import tensorflow as tf
from tensorflow.keras.layers import Input, MaxPool2D, Bidirectional, Reshape, GRU, Dense
from layers import ConvBnRelu


class CRNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.max_length = 24
        self.height, self.width = 432, 48
        self.model = self._build_model()
        self.num2char = tf.keras.layers.StringLookup(
            vocabulary = open('./assets/vocab.txt', encoding='utf-8').read().splitlines(),
            mask_token = '[PAD]', 
            invert = True,
        )
        
        
    def _build_model(self):
        image_input = Input(shape=(self.height, self.width, 3), dtype='float32', name='image')
        x = ConvBnRelu(64, 3, name='block1_convbn')(image_input)
        x = MaxPool2D((2, 2), name='block1_pool')(x)
        
        x = ConvBnRelu(128, 3, name='block2_convbn')(x)
        x = MaxPool2D((2, 2), name='block2_pool')(x)
        
        x = ConvBnRelu(256, 3, name='block3_convbn1')(x)
        x = ConvBnRelu(256, 3, name='block3_convbn2')(x)
        x = MaxPool2D((2, 2), name='block3_pool')(x)
        
        x = ConvBnRelu(512, 3, name='block4_convbn1')(x)
        x = ConvBnRelu(512, 3, name='block4_convbn2')(x)
        x = MaxPool2D((2, 2), name='block4_pool')(x)
        
        x = ConvBnRelu(512, 2, padding='valid', name='block5_convbn1')(x)
        x = ConvBnRelu(512, 2, padding='valid', name='block5_convbn2')(x)
        
        # Reshape accordingly before passing output to RNN
        _, height, width, channel = x.get_shape()
        feature_maps = Reshape(target_shape=((height, width * channel)), name='rnn_input')(x)
        
        # RNN layers
        bigru1 = Bidirectional(GRU(256, return_sequences=True), name='bigru1')(feature_maps)
        bigru2 = Bidirectional(GRU(256, return_sequences=True), name='bigru2')(bigru1)
        
        # Output layer
        y_pred = Dense(7482, activation='softmax', name='rnn_output')(bigru2)
        return tf.keras.Model(inputs=image_input, outputs=y_pred, name='CRNN')


    def distortion_free_resize(self, image, align_top=True):
        image = tf.image.resize(image, size=(self.height, self.width), preserve_aspect_ratio=True)
        pad_height = self.height - tf.shape(image)[0]
        pad_width = self.width - tf.shape(image)[1]
        if pad_height == 0 and pad_width == 0: return image

        # Only necessary if you want to do same amount of padding on both sides.
        if pad_height % 2 != 0:
            height = pad_height // 2
            pad_height_top, pad_height_bottom = height + 1, height
        else:
            pad_height_top = pad_height_bottom = pad_height // 2

        if pad_width % 2 != 0:
            width = pad_width // 2
            pad_width_left, pad_width_right = width + 1, width
        else:
            pad_width_left = pad_width_right = pad_width // 2

        return tf.pad(image, paddings=[
            [0, pad_height_top + pad_height_bottom] if align_top else [pad_height_top, pad_height_bottom],
            [pad_width_left, pad_width_right],
            [0, 0],
        ], constant_values=255) # Pad with white color


    def process_image(self, image, img_align_top=True):
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = self.distortion_free_resize(image, img_align_top)
        image = tf.cast(image, tf.float32) / 255.0
        return image

    
    def ctc_decode(self, predictions, max_length):
        input_length = tf.ones(len(predictions)) * predictions.shape[1]
        preds_decoded = tf.keras.backend.ctc_decode(
            predictions,
            input_length = input_length,
            greedy = True,
        )[0][0][:, :max_length]
        
        return tf.where(
            preds_decoded == tf.cast(1, tf.int64),
            tf.cast(-1, tf.int64), # Treat [UNK] token same as blank label
            preds_decoded
        )
        
        
    def tokens2texts(self, batch_tokens):
        batch_texts = []
        batch_tokens = self.ctc_decode(batch_tokens, self.max_length)

        for tokens in batch_tokens:
            indices = tf.gather(tokens, tf.where(tf.logical_and(tokens != 0, tokens != -1)))
            text = tf.strings.reduce_join(self.num2char(indices)) 
            text = text.numpy().decode('utf-8')
            batch_texts.append(text)
        return batch_texts 


    def predict_one_patch(self, patch_img):
        image = self.process_image(patch_img)
        pred_tokens = self.model.predict(tf.expand_dims(image, axis=0))
        pred_labels = self.tokens2texts(pred_tokens)
        return pred_labels[0]