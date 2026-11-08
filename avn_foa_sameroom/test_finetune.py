import unittest
import numpy as np
import torch
from .finetune import yaw, balanced_adpit_loss
from .loader import masked_adpit_loss

class YawTests(unittest.TestCase):
    def test_front_becomes_right_on_left_turn(self):
        wave=torch.zeros(4,32);wave[0]=1;wave[3]=1
        target=torch.zeros(10,6,4,21);target[:,0,0,4]=1;target[:,0,1,4]=1
        a,b=yaw(wave,target,np.pi/2)
        torch.testing.assert_close(a[0],wave[0]);torch.testing.assert_close(a[1],-wave[3],atol=1e-6,rtol=0)
        torch.testing.assert_close(b[:,0,2,4],-torch.ones(10),atol=1e-6,rtol=0)
        self.assertTrue(torch.equal(b[:,:,0],target[:,:,0]))
    def test_inverse_and_energy_and_all_tracks(self):
        g=torch.Generator().manual_seed(7);wave=torch.randn(4,256,generator=g)
        target=torch.randn(10,6,4,21,generator=g)
        a,b=yaw(wave,target,.73);x,y=yaw(a,b,-.73)
        torch.testing.assert_close(x,wave,atol=1e-6,rtol=1e-5);torch.testing.assert_close(y,target,atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(a[[1,2,3]].square().sum(0),wave[[1,2,3]].square().sum(0),atol=2e-6,rtol=1e-5)
        self.assertTrue(torch.equal(a[0],wave[0]));self.assertTrue(torch.equal(a[2],wave[2]))
class LossTests(unittest.TestCase):
    def test_standard_adpit_single_source_and_ignore(self):
        pred=torch.zeros(1,2,189,requires_grad=True);target=torch.zeros(1,2,6,4,21)
        target[0,0,0,0,3]=1;target[0,0,0,1,3]=1
        value=masked_adpit_loss(pred,target,torch.tensor([[True,False]]))
        self.assertAlmostEqual(float(value),1/63,places=7)
        value.backward();self.assertGreater(float(pred.grad[0,0].abs().sum()),0)
        self.assertTrue(torch.equal(pred.grad[0,1],torch.zeros(189)))
    def test_background_does_not_hide_foreground(self):
        pred=torch.zeros(1,2,189,requires_grad=True);target=torch.zeros(1,2,6,4,21)
        target[0,0,0,0,3]=1;target[0,0,0,1,3]=1
        value=balanced_adpit_loss(pred,target,torch.ones(1,2,dtype=torch.bool))
        self.assertAlmostEqual(float(value),1/6,places=6)
        value.backward();self.assertGreater(float(pred.grad.abs().sum()),0)
    def test_ignore_frames_have_zero_gradient(self):
        pred=torch.ones(1,2,189,requires_grad=True);target=torch.zeros(1,2,6,4,21)
        value=balanced_adpit_loss(pred,target,torch.tensor([[True,False]]));value.backward()
        self.assertTrue(torch.equal(pred.grad[0,1],torch.zeros(189)))

if __name__=='__main__':unittest.main()
