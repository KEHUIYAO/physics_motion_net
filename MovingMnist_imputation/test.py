class PhysicMotionNet:
    def __init__(self, physics_net):
        self.physics_net = physics_net

    def forward(self, images, image_mask):
        images = images.transpose(0, 1)
        image_mask = image_mask[0, :]
        if image_mask[0] == 0:  # the first time step should not be missing
            raise(ValueError("the first time step should not be missing"))
        T = len(image_mask)
        j = 0

        # feed in the first time step
        self.physics_net(images[j], True, False)

