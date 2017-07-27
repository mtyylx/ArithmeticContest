from keras.layers.merge import Concatenate
from keras.layers.core import Lambda
from keras.models import Model
import tensorflow as tf

# 用法
# 1. 建立模型: model = create_model(4, 32)
# 2. 转换模型：model = make_parallel(model, 2)   转换后的模型支持多GPU并行
# 3. 编译模型：model.compile(xxx)
# 4. 训练模型：model.fit_generator(xxx)
# 5. 测试模型：model.predict(xxx)
# 注：此方法为Data Parallelism的并行编程模式，即多个GPU训练的是同一个模型，但是灌的是不同的数据，通过调度来同时更新模型的参数。

def make_parallel(model, gpu_count):

    def get_slice(data, idx, parts):
        shape = tf.shape(data)
        size = tf.concat([shape[:1] // parts, shape[1:]], axis=0)
        stride = tf.concat([shape[:1] // parts, shape[1:] * 0], axis=0)
        start = stride * idx
        return tf.slice(data, start, size)

    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])

    # Place a copy of the model on each GPU, each getting a slice of the batch
    for i in range(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:

                inputs = []
                # Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'idx': i, 'parts': gpu_count})(x)
                    inputs.append(slice_n)

                outputs = model(inputs)

                if not isinstance(outputs, list):
                    outputs = [outputs]

                # Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])

    # merge outputs on CPU
    # 也可以设为GPU，如果CPU负载已经很大的话
    with tf.device('/cpu:0'):
        merged = []
        for outputs in outputs_all:
            merged.append(Concatenate(axis=0)(outputs))
        return Model(model.inputs, merged)
