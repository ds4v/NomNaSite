import cv2
import tensorflow as tf
from tensorflow.keras.layers import Input, UpSampling2D, Add, Concatenate, Lambda
from keras_resnet.models import ResNet18
from layers import ConvBnRelu, DeConvMap
from processor import PostProcessor


class DBNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.model = self._build_model()
        self.post_processor = PostProcessor(min_box_score=0.5, max_candidates=1000)
        

    def _build_model(self, k=50):
        image_input = Input(shape=(None, None, 3), name='image')
        backbone = ResNet18(inputs=image_input, include_top=False)
        
        C2, C3, C4, C5 = backbone.outputs
        in2 = ConvBnRelu(256, 1, name='in2')(C2)
        in3 = ConvBnRelu(256, 1, name='in3')(C3)
        in4 = ConvBnRelu(256, 1, name='in4')(C4)
        in5 = ConvBnRelu(256, 1, name='in5')(C5)
        
        # The pyramid features are up-sampled to the same scale and cascaded to produce feature F
        out4 = UpSampling2D(2, name='up5')(in5)  + in4
        out3 = UpSampling2D(2, name='up4')(out4) + in3
        out2 = UpSampling2D(2, name='up3')(out3) + in2
        
        P5 = tf.keras.Sequential([ConvBnRelu(64, 3), UpSampling2D(8)], name='P5')(in5)
        P4 = tf.keras.Sequential([ConvBnRelu(64, 3), UpSampling2D(4)], name='P4')(out4)
        P3 = tf.keras.Sequential([ConvBnRelu(64, 3), UpSampling2D(2)], name='P3')(out3)
        P2 = ConvBnRelu(64, 3, name='P2')(out2)
        
        # Calculate DBNet maps
        fuse = Concatenate(name='fuse')([P2, P3, P4, P5]) # (batch_size, /4, /4, 256)
        binarize_map = DeConvMap(64, name='probability_map')(fuse)
        threshold_map = DeConvMap(64, name='threshold_map')(fuse)
        thresh_binary = Lambda( 
            lambda x: 1 / (1 + tf.exp(-k * (x[0] - x[1]))), # b_hat = 1 / (1 + e^(-k(P - T)))
            name = 'approximate_binary_map'
        )([binarize_map, threshold_map]) 
        
        return tf.keras.Model(
            inputs = image_input, 
            outputs = [binarize_map, threshold_map, thresh_binary], 
            name = 'DBNet'
        )


    def resize_image_short_side(self, image, image_short_side=736):
        height, width, _ = image.shape
        if height < width:
            new_height = image_short_side
            new_width = int(round(new_height / height * width / 32) * 32)
        else:
            new_width = image_short_side
            new_height = int(round(new_width / width * height / 32) * 32)
        return cv2.resize(image, (new_width, new_height))


    def predict_one_page(self, page_path):
        raw_image = cv2.cvtColor(cv2.imread(page_path), cv2.COLOR_BGR2RGB)
        image = self.resize_image_short_side(raw_image)
        image = image.astype(float) / 255.0

        binarize_map, _, _ = self.model(tf.expand_dims(image, 0), training=False)
        batch_boxes, batch_scores = self.post_processor(binarize_map.numpy(), [raw_image.shape[:2]])
        return raw_image, batch_boxes[0], batch_scores[0]
