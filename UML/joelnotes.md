# test_quant.py

1. Create model

2. model.to(device) + model.eval()

3. data work

4. Quantization

5. calibration data = get first 10 batches of training data 

6. call model.model_open_calibrate()
    All modules of type [QConv2d, QLinear, QAct, QIntSoftmax] gets calibrate = True

7. For each batch of calibration data: model(batch of images) feed forward.
    There is quantization method inside of model.forward()
    But attribute module.quant = false by default so it does not quantize during calibration.

7.2 If at the large batch, call model.model_open_last_calibrate()
    All modules of type [QConv2d, QLinear, QAct, QIntSoftmax] gets last_calibrate = True
    (this is used for the omse calibration method, which needs to know the last batch of data)

8. model.model_close_calibrate()
    All modules of type [QConv2d, QLinear, QAct, QIntSoftmax] gets calibrate = False

9. model.model_quant()
    Sets quant = True (for all modules of type [QConv2d, QLinear, QAct, QIntSoftmax])
    Sets quant = 'int' (for all modules of type [QIntLayerNorm] if self.cfg.INT_NORM is True)

## Call stack
The main method calls the method of vit_quant.py, swin_transformer.py, and qmodules.py depending on which model is used.
During calibration, the forward method is called but quantization is turned off.
During validation forward pass, the quantization is turned on.
From the model.forward() method we call

        if self.input_quant:
            x = self.qact_input(x)

        

# Vit_quant.py - defines calibration

    def model_open_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                m.calibrate = True

    def model_open_last_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                m.last_calibrate = True

    def model_close_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                m.calibrate = False

# Vit_quant.py - defines quantization as well

    def model_quant(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                m.quant = True
            if self.cfg.INT_NORM:
                if type(m) in [QIntLayerNorm]:
                    m.mode = 'int'

    def model_dequant(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                m.quant = False

