class DECAConfig:
    def __init__(self):
        self.inputpath = "TestSamples/examples"
        self.savefolder = "TestSamples/examples/results"
        self.device = "cuda"
        self.iscrop = True
        self.sample_step = 10
        self.detector = "fan"
        self.rasterizer_type = "standard"
        self.render_orig = True
        self.useTex = False
        self.extractTex = True
        self.saveVis = True
        self.saveKpt = False
        self.saveDepth = False
        self.saveObj = False
        self.saveMat = False
        self.saveImages = False