# this is an implementation of the method described in 
# 
# Neural Algorithm of Artistic Style. Gatys, Ecker, Bethge 2015 http://arxiv.org/pdf/1508.06576.pdf
#
# this code is meant as an executable sketch showing the kinds of computation that need to be done. 
# running the code makes the most sense in an interactive environment, eg. within an ipython shell 


import numpy
import numpy.random
numpy_rng = numpy.random.RandomState(1)
import pylab
import theano 
import sklearn_theano
import sklearn_theano.feature_extraction
import sklearn_theano.utils 
from scipy import ndimage 


network = "googlenet" #"overfeat" or "googlenet"
contentimagefile = "./contentimage.png"
styleimagefile = "./styleimage.jpg"



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


#LOAD IMAGE DATA 
contentimage = getimage(contentimagefile)[numpy.newaxis,:,:,:]
styleimage = getimage(styleimagefile)[numpy.newaxis,:,:,:]


#GET CONVNET LAYERS IN THE FORM OF THEANO EXPRESSIONS: 
if network == "overfeat":
    arch = sklearn_theano.feature_extraction.overfeat._get_architecture(large_network=False, detailed=True)
    #arch = sklearn_theano.feature_extraction.overfeat._get_architecture(large_network=True, detailed=True)
    allexpressions, input_var = sklearn_theano.base.fuse(arch, output_expressions=range(22)) 
    expressions = [allexpressions[4], allexpressions[8], allexpressions[11]]
    contentweights, styleweights = [0.0, 1.0, 0.0], [10.0, 100.0, 1000.0] 
    #layers, contentweights, styleweights = [2, 5, 9, 12, 15, 18], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 10.0, 100.0, 1000.0] 
elif network == "googlenet":
    theanolayers, input_var = sklearn_theano.feature_extraction.caffe.googlenet.create_theano_expressions()
    expressions = [theanolayers['conv1/7x7_s2'], theanolayers['conv2/3x3'], theanolayers['inception_3a/output'], theanolayers['inception_3b/output'], theanolayers['inception_4a/output'], theanolayers['inception_4b/output'], theanolayers['inception_4d/output'], theanolayers['inception_4e/output']]
    contentweights, styleweights = [0.0, 0.0, 0.00001, 0.0, 0.00001, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] 
    #contentweights, styleweights = [0.0, 0.0, 0.00001, 0.0, 0.00001, 0.0, 0.0, 0.0, 0.0], [0.1, 0.1, 1.0, 1.0, 10.0, 10.0, 100.0, 100.0] 

styleweights = [theano.shared(value=w, name=str(i)) for i, w in enumerate(styleweights)]
contentweights = [theano.shared(value=w, name=str(i)) for i, w in enumerate(contentweights)]


#EXTRACT LAYER ACTIVATIONS AND DEFINE CONTENT AND STYLE COSTS: 
totalcost = 0.0
layer_content_cost = []
layer_style_cost = []
#layer_content_cost_function = []
#layer_style_cost_function = []

for layerindex, expression in enumerate(expressions):
    layeroutput_function = theano.function([input_var], expression)
    contentimage_layer = layeroutput_function(contentimage)
    styleimage_layer = layeroutput_function(styleimage)

    #DEFINE COST AND GRADIENT FOR CONTENT: 
    layer_content_cost.append(((expression-contentimage_layer)**2).sum())
    #layer_content_cost_function.append(theano.function([input_var], layer_content_cost[-1]))

    #DEFINE COST AND GRADIENT FOR STYLE (WE USE SCAN TO ITERATE OVER THE ROWS OF THE FEATUREMAP, OTHER ATTEMPTS LED TO RECURSION_DEPTH ISSUES):
    grammian_original = numpy.zeros((styleimage_layer.shape[1], styleimage_layer.shape[1]), dtype="float32")
    for i in range(styleimage_layer.shape[1]):
        for j in range(styleimage_layer.shape[1]):
            grammian_original[i,j] = (styleimage_layer[0,i,:,:] * styleimage_layer[0,j,:,:]).sum(1).sum(0)
    
    grammian_testimage, updates = theano.scan(fn=lambda onefeatureplane, featuremap: (onefeatureplane.dimshuffle('x',0,1)*featuremap).sum(2).sum(1),
                                      outputs_info=None,
                                      sequences=[expression[0,:,:,:]], 
                                      non_sequences=expression[0,:,:,:])
    
    layer_style_cost.append(((grammian_testimage - grammian_original)**2).sum() / (2*(styleimage_layer.shape[2]*styleimage_layer.shape[3])**2 * (styleimage_layer.shape[1])**2))
    #layer_style_cost_function.append(theano.function([input_var], layer_style_cost[-1]))

    #DEFINE TOTAL COST AS WEIGHTED SUM OF CONTENT AND STYLE COST
    totalcost += contentweights[layerindex] * layer_content_cost[layerindex] + styleweights[layerindex] * layer_style_cost[layerindex]

totalgrad = theano.grad(totalcost, input_var)

#COMPILE THEANO FUNCTIONS: 
cost = theano.function([input_var], totalcost)
grad = theano.function([input_var], totalgrad)

#CONJGRAD BASED OPTIMIZATION FOR POTENTIALLY FASTER OPTIMIZATION (REQUIRES minimize.py): 
def conjgrad(im, maxnumlinesearch=10, imshape=styleimage.shape):
    import minimize
    im_flat, fs, numlinesearches = minimize.minimize(im.flatten(), lambda x: cost(x.reshape(imshape)), lambda x: grad(x.reshape(imshape)).flatten(), args=[], maxnumlinesearch=maxnumlinesearch, verbose=False)
    return im_flat.reshape(imshape)



#TRY IT OUT:

#INFERENCE WITH GRADIENT DESCENT:
#imout = numpy_rng.randint(256, size=(contentimage.shape)).astype("float32")
#imout = 128 * numpy.ones(contentimage.shape).astype("float32")
imout = contentimage.copy()
stepsize = 0.1
momentum = 0.9
inc = 0
for i in range(100):
    inc = momentum * inc - stepsize * grad(imout)
    imout += inc 
    imout[imout<10] = 10
    imout[imout>245] = 245
    print cost(imout)


#INFERENCE WITH CONJUGATE GRADIENTS:
#conjgrad(imout)


#SHOW RESULT:
showim(imout)


print """to continue training, use something like: 
for i in range(100):
    inc = momentum * inc - stepsize * grad(imout)
    imout += inc 
    imout[imout<10] = 10
    imout[imout>245] = 245
    print cost(imout)


or 

imout = conjgrad(imout) 
"""



