import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Activation, Conv2D, Dense, DepthwiseConv2D, Dropout, GlobalAveragePooling2D, BatchNormalization
from keras.models import Model

def _conv_block(inputs, filters, kernel=(3, 3), strides=(1, 1)):
    x = Conv2D(filters, kernel,
               padding='same',
               use_bias=False,
               strides=strides,
               name='conv1')(inputs)
    x = BatchNormalization(name='conv1_bn')(x)
    return Activation(relu6, name='conv1_relu')(x)

def _depthwise_conv_block(inputs, pointwise_conv_filters, depth_multiplier=1, strides=(1, 1), block_id=1):
    x = DepthwiseConv2D((3, 3),
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id)(inputs)

    x = BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)

def relu6(x):
    return K.relu(x, max_value=6)

def MobileNet(inputs, embedding_size=128, dropout_keep_prob=0.4, alpha=1.0, depth_multiplier=1):

    x = _conv_block(inputs, 32, strides=(2, 2))

    x = _depthwise_conv_block(x, 64, depth_multiplier, block_id=1)

    x = _depthwise_conv_block(x, 128, depth_multiplier, strides=(2, 2), block_id=2)
    x = _depthwise_conv_block(x, 128, depth_multiplier, block_id=3)

    x = _depthwise_conv_block(x, 256, depth_multiplier, strides=(2, 2), block_id=4)
    x = _depthwise_conv_block(x, 256, depth_multiplier, block_id=5)

    x = _depthwise_conv_block(x, 512, depth_multiplier, strides=(2, 2), block_id=6)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=7)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=8)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=9)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=10)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=11)

    x = _depthwise_conv_block(x, 1024, depth_multiplier, strides=(2, 2), block_id=12)
    x = _depthwise_conv_block(x, 1024, depth_multiplier, block_id=13)

    x = GlobalAveragePooling2D()(x)

    x = Dropout(1.0 - dropout_keep_prob, name='Dropout')(x)

    x = Dense(embedding_size, use_bias=False, name='Bottleneck')(x)
    x = BatchNormalization(momentum=0.995, epsilon=0.001, scale=False,
                           name='BatchNorm_Bottleneck')(x)

    model = Model(inputs, x, name='mobilenet')

    return model

def facenet(input_shape, num_classes = None, mode = "train"):
    inputs = Input(shape=input_shape)
    model = MobileNet(inputs, dropout_keep_prob = 0.5)

    if mode == "train":
        logits          = Dense(num_classes)(model.output)
        softmax         = Activation("softmax", name = "Softmax")(logits)
        
        normalize       = Lambda(lambda  x: K.l2_normalize(x, axis=1), name="Embedding")(model.output)
        combine_model   = Model(inputs, [softmax, normalize])
        return combine_model
    elif mode == "predict":
        x = Lambda(lambda  x: K.l2_normalize(x, axis=1), name="Embedding")(model.output)
        model = Model(inputs,x)
        return model
    else:
        raise ValueError('Unsupported mode - `{}`, Use train, predict.'.format(mode))

def triplet_loss(alpha = 0.2, batch_size = 32):
    def _triplet_loss(y_true, y_pred):
        anchor, positive, negative = y_pred[:batch_size], y_pred[batch_size:int(2 * batch_size)], y_pred[-batch_size:]

        pos_dist    = K.sqrt(K.sum(K.square(anchor - positive), axis=-1))
        neg_dist    = K.sqrt(K.sum(K.square(anchor - negative), axis=-1))

        basic_loss  = pos_dist - neg_dist + alpha
        
        idxs        = tf.where(basic_loss > 0)
        select_loss = tf.gather_nd(basic_loss, idxs)

        loss        = K.sum(K.maximum(basic_loss, 0)) / tf.cast(tf.maximum(1, tf.shape(select_loss)[0]), tf.float32)
        return loss
    return _triplet_loss