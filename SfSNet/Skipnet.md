# Enter your network definition here.
# Use Shift+Enter to update the visualization.

#SkipNet

name : "SkipNet"
#data


layer {
  name: "data"
  type: "Input"
  top: "data"
    input_param { shape: { dim: 1 dim: 3 dim: 128 dim: 128 } }

}

############################ Initial
#C64
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  param {
name : "c1_w"
    lr_mult: 1
    decay_mult: 1
  }
  param {
name : "c1_b"
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 64
    kernel_size: 4
    stride: 2
    pad: 3
    weight_filler {
      type: "xavier"
    }
  }
}

layer {
  name: "relu1"
  type: "ReLU"
  bottom: "conv1"
  top: "conv1"
}

#C128
layer {
  name: "conv2"
  type: "Convolution"
  bottom: "conv1"
  top: "conv2"
  param {
name: "c2_w"
    lr_mult: 1
    decay_mult: 1
  }
  param {
name: "c2_b"
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 128
    kernel_size: 4
    stride: 1
    pad: 1
    weight_filler {
      type: "xavier"
    }
  }
}

layer {
  name: "bn2"
  type: "BatchNorm"
  bottom: "conv2"
  top: "conv2"
  batch_norm_param {
    use_global_stats: false
  }
  param {
name : "b2_a"
    lr_mult: 0
  }
  param {
name : "b2_b"
    lr_mult: 0
  }
  param {
name: "b2_c"
    lr_mult: 0
  }
  include {
    phase: TRAIN
  }
}
layer {
  name: "bn2"
  type: "BatchNorm"
  bottom: "conv2"
  top: "conv2"
  batch_norm_param {
    use_global_stats: true
  }
  param {
name : "b2_a"
    lr_mult: 0
  }
  param {
name : "b2_b"
    lr_mult: 0
  }
  param {
name: "b2_c"
    lr_mult: 0
  }
  include {
    phase: TEST
  }
}

layer {
  name: "relu2"
  type: "ReLU"
  bottom: "conv2"
  top: "conv2"
}

#C256 S3
layer {
  name: "conv3"
  type: "Convolution"
  bottom: "conv2"
  top: "conv3"
  param {
name : "c3_w"
    lr_mult: 1
    decay_mult: 1
  }
  param {
name : "c3_b"
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 256
    kernel_size: 4
    stride: 2
    pad: 1
    weight_filler {
      type: "xavier"
    }
  }
}

layer {
  name: "bn3"
  type: "BatchNorm"
  bottom: "conv3"
  top: "conv3"
  batch_norm_param {
    use_global_stats: false
  }
  param {
name : "b2_a"
    lr_mult: 0
  }
  param {
name : "b2_b"
    lr_mult: 0
  }
  param {
name: "b2_c"
    lr_mult: 0
  }
  include {
    phase: TRAIN
  }
}

layer {
  name: "bn3"
  type: "BatchNorm"
  bottom: "conv3"
  top: "conv3"
  batch_norm_param {
    use_global_stats: true
  }
  param {
name : "b2_a"
    lr_mult: 0
  }
  param {
name : "b2_b"
    lr_mult: 0
  }
  param {
name: "b2_c"
    lr_mult: 0
  }
  include {
    phase: TEST
  }
}

layer {
  name: "relu3"
  type: "ReLU"
  bottom: "conv3"
  top: "conv3"
}

#C256 S4
layer {
  name: "conv4"
  type: "Convolution"
  bottom: "conv3"
  top: "conv4"
  param {
name : "c3_w"
    lr_mult: 1
    decay_mult: 1
  }
  param {
name : "c3_b"
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 256
    kernel_size: 4
    stride: 2
    pad: 1
    weight_filler {
      type: "xavier"
    }
  }
}

layer {
  name: "bn4"
  type: "BatchNorm"
  bottom: "conv4"
  top: "conv4"
  batch_norm_param {
    use_global_stats: false
  }
  param {
name : "b2_a"
    lr_mult: 0
  }
  param {
name : "b2_b"
    lr_mult: 0
  }
  param {
name: "b2_c"
    lr_mult: 0
  }
  include {
    phase: TRAIN
  }
}

layer {
  name: "bn4"
  type: "BatchNorm"
  bottom: "conv4"
  top: "conv4"
  batch_norm_param {
    use_global_stats: true
  }
  param {
name : "b2_a"
    lr_mult: 0
  }
  param {
name : "b2_b"
    lr_mult: 0
  }
  param {
name: "b2_c"
    lr_mult: 0
  }
  include {
    phase: TEST
  }
}

layer {
  name: "relu4"
  type: "ReLU"
  bottom: "conv4"
  top: "conv4"
}

#C256 S5
layer {
  name: "conv5"
  type: "Convolution"
  bottom: "conv4"
  top: "conv5"
  param {
name : "c3_w"
    lr_mult: 1
    decay_mult: 1
  }
  param {
name : "c3_b"
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 256
    kernel_size: 4
    stride: 2
    pad: 1
    weight_filler {
      type: "xavier"
    }
  }
}

layer {
  name: "bn5"
  type: "BatchNorm"
  bottom: "conv5"
  top: "conv5"
  batch_norm_param {
    use_global_stats: false
  }
  param {
name : "b2_a"
    lr_mult: 0
  }
  param {
name : "b2_b"
    lr_mult: 0
  }
  param {
name: "b2_c"
    lr_mult: 0
  }
  include {
    phase: TRAIN
  }
}

layer {
  name: "bn5"
  type: "BatchNorm"
  bottom: "conv5"
  top: "conv5"
  batch_norm_param {
    use_global_stats: true
  }
  param {
name : "b2_a"
    lr_mult: 0
  }
  param {
name : "b2_b"
    lr_mult: 0
  }
  param {
name: "b2_c"
    lr_mult: 0
  }
  include {
    phase: TEST
  }
}

layer {
  name: "relu5"
  type: "ReLU"
  bottom: "conv5"
  top: "conv5"
}

layer {
name: "fc_layer"
type: "InnerProduct"
bottom: "conv5"
top: "fc_layer"
param {
lr_mult: 1
decay_mult: 1
}
param {
lr_mult: 2
decay_mult: 0
}
inner_product_param {
num_output: 256
weight_filler {
type: "gaussian"
std: 0.005
}
bias_filler {
type: "constant"
value: 1
}
}
}

layer {
name: "MLP1_Normal"
type: "InnerProduct"
bottom: "fc_layer"
top: "MLP1_Normal"
param {
lr_mult: 1
decay_mult: 1
}
param {
lr_mult: 2
decay_mult: 0
}
inner_product_param {
num_output: 256
weight_filler {
type: "gaussian"
std: 0.005
}
bias_filler {
type: "constant"
value: 1
}
}
}

#Deconv1
layer {
  name: "nup1"
  type: "Deconvolution"
  bottom: "MLP1_Normal" 
  top: "nup1"
  convolution_param {
    kernel_size: 4
    stride: 2
    num_output: 256
    group: 256
    pad: 1
    weight_filler { 
       type: "bilinear" 
    } 
    bias_term: false
  }
  param { 
    lr_mult: 0 
    decay_mult: 0 
  }
}

layer {
  name: "bn_up1"
  type: "BatchNorm"
  bottom: "nup1"
  top: "nup1"
  batch_norm_param {
    use_global_stats: false
  }
  param {
name : "b2_a"
    lr_mult: 0
  }
  param {
name : "b2_b"
    lr_mult: 0
  }
  param {
name: "b2_c"
    lr_mult: 0
  }
  include {
    phase: TRAIN
  }
}


layer {
  name: "relu_up1"
  type: "ReLU"
  bottom: "nup1"
  top: "nup1"
}

layer {
  name: "nsum1"
  type: "Eltwise"
  bottom: "nup1"
  bottom: "conv1"
  top: "nsum1"
  eltwise_param { operation: SUM }
}

#Deconv2
layer {
  name: "nup2"
  type: "Deconvolution"
  bottom: "nsum1" 
  top: "nup2"
  convolution_param {
    kernel_size: 4
    stride: 2
    num_output: 256
    group: 256
    pad: 1
    weight_filler { 
       type: "bilinear" 
    } 
    bias_term: false
  }
  param { 
    lr_mult: 0 
    decay_mult: 0 
  }
}

layer {
  name: "bn_up2"
  type: "BatchNorm"
  bottom: "nup2"
  top: "nup2"
  batch_norm_param {
    use_global_stats: false
  }
  param {
name : "b2_a"
    lr_mult: 0
  }
  param {
name : "b2_b"
    lr_mult: 0
  }
  param {
name: "b2_c"
    lr_mult: 0
  }
  include {
    phase: TRAIN
  }
}


layer {
  name: "relu_up2"
  type: "ReLU"
  bottom: "nup2"
  top: "nup2"
}

layer {
  name: "nsum2"
  type: "Eltwise"
  bottom: "nup2"
  bottom: "conv2"
  top: "nsum2"
  eltwise_param { operation: SUM }
}

#Deconv3
layer {
  name: "nup3"
  type: "Deconvolution"
  bottom: "nsum2" 
  top: "nup3"
  convolution_param {
    kernel_size: 4
    stride: 2
    num_output: 256
    group: 256
    pad: 1
    weight_filler { 
       type: "bilinear" 
    } 
    bias_term: false
  }
  param { 
    lr_mult: 0 
    decay_mult: 0 
  }
}

layer {
  name: "bn_up3"
  type: "BatchNorm"
  bottom: "nup3"
  top: "nup3"
  batch_norm_param {
    use_global_stats: false
  }
  param {
name : "b2_a"
    lr_mult: 0
  }
  param {
name : "b2_b"
    lr_mult: 0
  }
  param {
name: "b2_c"
    lr_mult: 0
  }
  include {
    phase: TRAIN
  }
}


layer {
  name: "relu_up3"
  type: "ReLU"
  bottom: "nup3"
  top: "nup3"
}

layer {
  name: "nsum3"
  type: "Eltwise"
  bottom: "nup3"
  bottom: "conv3"
  top: "nsum3"
  eltwise_param { operation: SUM }
}

#Deconv4
layer {
  name: "nup4"
  type: "Deconvolution"
  bottom: "nsum3" 
  top: "nup4"
  convolution_param {
    kernel_size: 4
    stride: 2
    num_output: 128
    group: 128
    pad: 1
    weight_filler { 
       type: "bilinear" 
    } 
    bias_term: false
  }
  param { 
    lr_mult: 0 
    decay_mult: 0 
  }
}

layer {
  name: "bn_up4"
  type: "BatchNorm"
  bottom: "nup4"
  top: "nup4"
  batch_norm_param {
    use_global_stats: false
  }
  param {
name : "b2_a"
    lr_mult: 0
  }
  param {
name : "b2_b"
    lr_mult: 0
  }
  param {
name: "b2_c"
    lr_mult: 0
  }
  include {
    phase: TRAIN
  }
}


layer {
  name: "relu_up4"
  type: "ReLU"
  bottom: "nup4"
  top: "nup4"
}

layer {
  name: "nsum4"
  type: "Eltwise"
  bottom: "nup4"
  bottom: "conv4"
  top: "nsum4"
  eltwise_param { operation: SUM }
}


#Deconv5
layer {
  name: "nup5"
  type: "Deconvolution"
  bottom: "nsum4" 
  top: "nup5"
  convolution_param {
    kernel_size: 4
    stride: 2
    num_output: 64
    group: 64
    pad: 1
    weight_filler { 
       type: "bilinear" 
    } 
    bias_term: false
  }
  param { 
    lr_mult: 0 
    decay_mult: 0 
  }
}

layer {
  name: "bn_up5"
  type: "BatchNorm"
  bottom: "nup5"
  top: "nup5"
  batch_norm_param {
    use_global_stats: false
  }
  param {
name : "b2_a"
    lr_mult: 0
  }
  param {
name : "b2_b"
    lr_mult: 0
  }
  param {
name: "b2_c"
    lr_mult: 0
  }
  include {
    phase: TRAIN
  }
}


layer {
  name: "relu_up5"
  type: "ReLU"
  bottom: "nup5"
  top: "nup5"
}

layer {
  name: "nsum5"
  type: "Eltwise"
  bottom: "nup5"
  bottom: "conv5"
  top: "nsum5"
  eltwise_param { operation: SUM }
}

#C*3
layer {
  name: "Nconv0"
  type: "Convolution"
  bottom: "nsum5"
  top: "Nconv0"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 3
    kernel_size: 1
    stride: 1
    pad: 0
    weight_filler {
      type: "xavier"
    }
  }
}




layer {
name: "MLP2_Albedo"
type: "InnerProduct"
bottom: "fc_layer"
top: "MLP2_Albedo"
param {
lr_mult: 1
decay_mult: 1
}
param {
lr_mult: 2
decay_mult: 0
}
inner_product_param {
num_output: 256
weight_filler {
type: "gaussian"
std: 0.005
}
bias_filler {
type: "constant"
value: 1
}
}
}

#Deconv1
layer {
  name: "aup1"
  type: "Deconvolution"
  bottom: "MLP2_Albedo" 
  top: "aup1"
  convolution_param {
    kernel_size: 4
    stride: 2
    num_output: 256
    group: 256
    pad: 1
    weight_filler { 
       type: "bilinear" 
    } 
    bias_term: false
  }
  param { 
    lr_mult: 0 
    decay_mult: 0 
  }
}

layer {
  name: "bn_aup1"
  type: "BatchNorm"
  bottom: "aup1"
  top: "aup1"
  batch_norm_param {
    use_global_stats: false
  }
  param {
name : "b2_a"
    lr_mult: 0
  }
  param {
name : "b2_b"
    lr_mult: 0
  }
  param {
name: "b2_c"
    lr_mult: 0
  }
  include {
    phase: TRAIN
  }
}


layer {
  name: "relu_aup1"
  type: "ReLU"
  bottom: "aup1"
  top: "aup1"
}

layer {
  name: "asum1"
  type: "Eltwise"
  bottom: "aup1"
  bottom: "conv1"
  top: "asum1"
  eltwise_param { operation: SUM }
}

#Deconv2
layer {
  name: "aup2"
  type: "Deconvolution"
  bottom: "asum1" 
  top: "aup2"
  convolution_param {
    kernel_size: 4
    stride: 2
    num_output: 256
    group: 256
    pad: 1
    weight_filler { 
       type: "bilinear" 
    } 
    bias_term: false
  }
  param { 
    lr_mult: 0 
    decay_mult: 0 
  }
}

layer {
  name: "bn_aup2"
  type: "BatchNorm"
  bottom: "aup2"
  top: "aup2"
  batch_norm_param {
    use_global_stats: false
  }
  param {
name : "b2_a"
    lr_mult: 0
  }
  param {
name : "b2_b"
    lr_mult: 0
  }
  param {
name: "b2_c"
    lr_mult: 0
  }
  include {
    phase: TRAIN
  }
}


layer {
  name: "relu_aup2"
  type: "ReLU"
  bottom: "aup2"
  top: "aup2"
}

layer {
  name: "asum2"
  type: "Eltwise"
  bottom: "aup2"
  bottom: "conv2"
  top: "asum2"
  eltwise_param { operation: SUM }
}

#Deconv3
layer {
  name: "aup3"
  type: "Deconvolution"
  bottom: "asum2" 
  top: "aup3"
  convolution_param {
    kernel_size: 4
    stride: 2
    num_output: 256
    group: 256
    pad: 1
    weight_filler { 
       type: "bilinear" 
    } 
    bias_term: false
  }
  param { 
    lr_mult: 0 
    decay_mult: 0 
  }
}

layer {
  name: "bn_aup3"
  type: "BatchNorm"
  bottom: "aup3"
  top: "aup3"
  batch_norm_param {
    use_global_stats: false
  }
  param {
name : "b2_a"
    lr_mult: 0
  }
  param {
name : "b2_b"
    lr_mult: 0
  }
  param {
name: "b2_c"
    lr_mult: 0
  }
  include {
    phase: TRAIN
  }
}


layer {
  name: "relu_aup3"
  type: "ReLU"
  bottom: "aup3"
  top: "aup3"
}

layer {
  name: "asum3"
  type: "Eltwise"
  bottom: "aup3"
  bottom: "conv3"
  top: "asum3"
  eltwise_param { operation: SUM }
}

#Deconv4
layer {
  name: "aup4"
  type: "Deconvolution"
  bottom: "asum3" 
  top: "aup4"
  convolution_param {
    kernel_size: 4
    stride: 2
    num_output: 128
    group: 128
    pad: 1
    weight_filler { 
       type: "bilinear" 
    } 
    bias_term: false
  }
  param { 
    lr_mult: 0 
    decay_mult: 0 
  }
}

layer {
  name: "bn_aup4"
  type: "BatchNorm"
  bottom: "anup4"
  top: "aup4"
  batch_norm_param {
    use_global_stats: false
  }
  param {
name : "b2_a"
    lr_mult: 0
  }
  param {
name : "b2_b"
    lr_mult: 0
  }
  param {
name: "b2_c"
    lr_mult: 0
  }
  include {
    phase: TRAIN
  }
}


layer {
  name: "relu_aup4"
  type: "ReLU"
  bottom: "aup4"
  top: "aup4"
}

layer {
  name: "asum4"
  type: "Eltwise"
  bottom: "aup4"
  bottom: "conv4"
  top: "asum4"
  eltwise_param { operation: SUM }
}


#Deconv5
layer {
  name: "aup5"
  type: "Deconvolution"
  bottom: "asum4" 
  top: "aup5"
  convolution_param {
    kernel_size: 4
    stride: 2
    num_output: 64
    group: 64
    pad: 1
    weight_filler { 
       type: "bilinear" 
    } 
    bias_term: false
  }
  param { 
    lr_mult: 0 
    decay_mult: 0 
  }
}

layer {
  name: "bn_aup5"
  type: "BatchNorm"
  bottom: "aup5"
  top: "aup5"
  batch_norm_param {
    use_global_stats: false
  }
  param {
name : "b2_a"
    lr_mult: 0
  }
  param {
name : "b2_b"
    lr_mult: 0
  }
  param {
name: "b2_c"
    lr_mult: 0
  }
  include {
    phase: TRAIN
  }
}


layer {
  name: "relu_aup5"
  type: "ReLU"
  bottom: "aup5"
  top: "aup5"
}

layer {
  name: "asum5"
  type: "Eltwise"
  bottom: "aup5"
  bottom: "conv5"
  top: "asum5"
  eltwise_param { operation: SUM }
}

#C*3
layer {
  name: "Aconv0"
  type: "Convolution"
  bottom: "asum5"
  top: "Aconv0"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 3
    kernel_size: 1
    stride: 1
    pad: 0
    weight_filler {
      type: "xavier"
    }
  }
}

layer {
name: "MLP3_Light"
type: "InnerProduct"
bottom: "fc_layer"
top: "MLP3_Light"
param {
lr_mult: 1
decay_mult: 1
}
param {
lr_mult: 2
decay_mult: 0
}
inner_product_param {
num_output: 256
weight_filler {
type: "gaussian"
std: 0.005
}
bias_filler {
type: "constant"
value: 1
}
}
}

layer {
name: "FC_light"
type: "InnerProduct"
bottom: "MLP3_Light"
top: "FC_Light"
param {
lr_mult: 1
decay_mult: 1
}
param {
lr_mult: 2
decay_mult: 0
}
inner_product_param {
num_output: 27
weight_filler {
type: "gaussian"
std: 0.005
}
bias_filler {
type: "constant"
value: 1
}
}
}



