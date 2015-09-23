# this is an implementation of the method described in 
# 
# Neural Algorithm of Artistic Style. Gatys, Ecker, Bethge 2015 http://arxiv.org/pdf/1508.06576.pdf
# 
# see bottom of file for a usage example 


import numpy
import numpy.random
numpy_rng = numpy.random.RandomState(1)
import pylab
import theano 
import sklearn_theano
import sklearn_theano.feature_extraction
import sklearn_theano.utils 
from scipy import ndimage 


def showim(im):
    pylab.imshow((im.astype("float")/im.max())[0].transpose(1,2,0))


def getimage(filename):
    im = pylab.imread(filename)
    if len(im.shape) == 2: #got gray-value image?
         im = im[:,:,numpy.newaxis].repeat(3, axis=2)
    im = ndimage.zoom(im, [231./im.shape[0], 231./im.shape[1], 1])
    im = sklearn_theano.utils.check_tensor(im, dtype=numpy.float32, n_dim=4)
    min_size = (231, 231)
    x_midpoint = im.shape[2] // 2
    y_midpoint = im.shape[1] // 2
    x_lower_bound = x_midpoint - min_size[0] // 2
    if x_lower_bound <= 0:
        x_lower_bound = 0
    x_upper_bound = x_lower_bound + min_size[0]
    y_lower_bound = y_midpoint - min_size[1] // 2
    if y_lower_bound <= 0:
        y_lower_bound = 0
    y_upper_bound = y_lower_bound + min_size[1]
    crop_bounds_ = (x_lower_bound, x_upper_bound, y_lower_bound, y_upper_bound)
    im -= im.min(); im /= im.max(); im *= 255.
    return im[:, y_lower_bound:y_upper_bound, x_lower_bound:x_upper_bound, :].transpose(0, 3, 1, 2)[0,:,:,:]


class Imtransformer(object): 
    def __init__(self, contentimage, styleimage, network="googlenet", verbose=True):
        self.network = network

        #LOAD IMAGE DATA 
        if type(contentimage) == type(""):
            self.contentimage = getimage(contentimage)[numpy.newaxis,:,:,:] 
        else: 
            assert type(contentimage) == type(numpy.array([1])), "contentimage has to be string (filename) or ndarray (representing an image)" 
            self.contentimage = contentimage
        if type(styleimage) == type(""):
            self.styleimage = getimage(styleimage)[numpy.newaxis,:,:,:] 
        else: 
            assert type(styleimage) == type(numpy.array([1])), "styleimage has to be string (filename) or ndarray (representing an image)" 
            self.styleimage = styleimage

        #GET CONVNET LAYERS AS THEANO EXPRESSIONS: 
        if network == "overfeat":
            arch = sklearn_theano.feature_extraction.overfeat._get_architecture(large_network=False, detailed=True)
            #arch = sklearn_theano.feature_extraction.overfeat._get_architecture(large_network=True, detailed=True)
            allexpressions, input_var = sklearn_theano.base.fuse(arch, output_expressions=range(22)) 
            self.expressions = [allexpressions[4], allexpressions[8], allexpressions[11]]
            contentweights, styleweights = [0.0, 1.0, 0.0], [10.0, 100.0, 1000.0] 
            #layers, contentweights, styleweights = [2, 5, 9, 12, 15, 18], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 10.0, 100.0, 1000.0] 
        elif network == "googlenet":
            print "getting theano network" 
            theanolayers, self.input_var = sklearn_theano.feature_extraction.caffe.googlenet.create_theano_expressions()
            self.expressions = [theanolayers['conv1/7x7_s2'], theanolayers['conv2/3x3'], theanolayers['inception_3a/output'], theanolayers['inception_3b/output'], theanolayers['inception_4a/output'], theanolayers['inception_4b/output'], theanolayers['inception_4d/output'], theanolayers['inception_4e/output']]
            contentweights, styleweights = [0.0, 0.0, 0.01, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] 
            #contentweights, styleweights = [0.0, 0.0, 0.00001, 0.0, 0.00001, 0.0, 0.0, 0.0, 0.0], [0.1, 0.1, 1.0, 1.0, 10.0, 10.0, 100.0, 100.0] 

        self.contentweights = [theano.shared(value=w, name=str(i)) for i, w in enumerate(contentweights)]
        self.styleweights = [theano.shared(value=w, name=str(i)) for i, w in enumerate(styleweights)]

        #EXTRACT LAYER ACTIVATIONS AND DEFINE CONTENT AND STYLE COSTS: 
        self.totalcost = 0.0
        layer_content_cost = []
        layer_style_cost = []
        #layer_content_cost_function = []
        #layer_style_cost_function = []

        for layerindex, expression in enumerate(self.expressions):
            if verbose: print "compiling layer ", str(layerindex), " function"  
            layeroutput_function = theano.function([self.input_var], expression)

            if verbose: print "defining content cost for layer ", str(layerindex) 
            self.contentimage_layer = layeroutput_function(self.contentimage)
            self.styleimage_layer = layeroutput_function(self.styleimage)

            #DEFINE COST AND GRADIENT FOR CONTENT: 
            layer_content_cost.append(((expression-self.contentimage_layer)**2).sum())
            #layer_content_cost_function.append(theano.function([self.input_var], layer_content_cost[-1]))

            #DEFINE COST AND GRADIENT FOR STYLE (WE USE SCAN TO ITERATE OVER THE ROWS OF THE FEATUREMAP, OTHER ATTEMPTS LED TO RECURSION_DEPTH ISSUES):
            if verbose: print "defining style cost for layer ", str(layerindex) 
            grammian_original = numpy.zeros((self.styleimage_layer.shape[1], self.styleimage_layer.shape[1]), dtype="float32")
            for i in range(self.styleimage_layer.shape[1]):
                for j in range(self.styleimage_layer.shape[1]):
                    grammian_original[i,j] = (self.styleimage_layer[0,i,:,:] * self.styleimage_layer[0,j,:,:]).sum(1).sum(0)

            grammian_testimage, updates = theano.scan(fn=lambda onefeatureplane, featuremap: (onefeatureplane.dimshuffle('x',0,1)*featuremap).sum(2).sum(1),
                                              outputs_info=None,
                                              sequences=[expression[0,:,:,:]], 
                                              non_sequences=expression[0,:,:,:])

            layer_style_cost.append(((grammian_testimage - grammian_original)**2).sum() / (2*(self.styleimage_layer.shape[2]*self.styleimage_layer.shape[3])**2 * (self.styleimage_layer.shape[1])**2))
            #layer_style_cost_function.append(theano.function([self.input_var], layer_style_cost[-1]))

            #DEFINE TOTAL COST AS WEIGHTED SUM OF CONTENT AND STYLE COST
            self.totalcost += self.contentweights[layerindex] * layer_content_cost[layerindex] + self.styleweights[layerindex] * layer_style_cost[layerindex]

        self.totalgrad = theano.grad(self.totalcost, self.input_var)

        #COMPILE THEANO FUNCTIONS: 
        if verbose: print "compiling cost" 
        self.cost = theano.function([self.input_var], self.totalcost)
        if verbose: print "compiling grad" 
        self.grad = theano.function([self.input_var], self.totalgrad)

    def step(self, initimage=None, numsteps=100, optimizer="sgd", stepsize=0.1, momentum=0.9): 
        if initimage is None:
            initimage = numpy_rng.randint(256, size=(self.contentimage.shape)).astype("float32")

        assert type(initimage) == type(numpy.array([1])), "initimage has to be ndarray (representing an image)" 

        imout = initimage.copy() 

        if optimizer == "sgd": 
            inc = 0
            for i in range(numsteps):
                if i > 3:
                    inc = momentum * inc - stepsize * self.grad(imout)
                else: 
                    inc = inc - stepsize * self.grad(imout)
                imout += inc 
                imout[imout<10] = 10
                imout[imout>245] = 245
                print "cost ", self.cost(imout)
        elif optimizer == "conjgrad": 
            #CONJGRAD BASED OPTIMIZATION FOR POTENTIALLY FASTER OPTIMIZATION (REQUIRES minimize.py): 
            def conjgrad(im, maxnumlinesearch=numsteps, imshape=self.styleimage.shape):
                import minimize
                im_flat, fs, numlinesearches = minimize.minimize(im.flatten(), lambda x: self.cost(x.reshape(imshape)), lambda x: self.grad(x.reshape(imshape)).flatten(), args=[], maxnumlinesearch=maxnumlinesearch, verbose=False)
                return im_flat.reshape(imshape)
            imout = conjgrad(imout)
        else:
             assert False, "optimizer has to be 'sgd' or 'conjgrad'"

        return imout 


if __name__ == "__main__":
    transformer = Imtransformer("./contentimage.png", "./styleimage.png")
    transformedimage = transformer.step()
    showim(transformedimage) 
    #not good? let's try a few more iterations with larger stepsize: 
    transformedimage = transformer.step(transformedimage, numsteps=10, stepsize=0.3)
    showim(transformedimage) 


