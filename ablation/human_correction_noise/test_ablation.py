"""Scientific invariants for synthetic human noise and the optimized objective."""
import sys
from pathlib import Path
import unittest
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE.parents[1]/'src'))
from run import make_scenarios,objective
from fcl.auc_loss import soft_pairwise_auc
from fcl.train_auc_ce import auc_noise_only_a11_loss


class AblationTests(unittest.TestCase):
    def test_noise_invariants(self):
        truth=torch.arange(200)%5
        s=(torch.arange(200)%3==0).long()
        base=truth.clone();base[s==1]=(base[s==1]+1)%5
        levels=[1,.95,.9,.85,.8]
        result=make_scenarios(base,s,truth,5,levels,1042)
        again=make_scenarios(base,s,truth,5,levels,1042)
        previous=set()
        for level in levels:
            r=result[level];indices=set(r['corrupted_indices'].tolist())
            self.assertTrue(previous<=indices)
            self.assertTrue(torch.equal(r['s'],s))
            self.assertTrue(torch.equal(r['y_obs'][s==1],base[s==1]))
            self.assertTrue(torch.equal(r['y_obs'],again[level]['y_obs']))
            self.assertEqual(int(r['y_obs'][s==0].ne(truth[s==0]).sum()),round((1-level)*int(s.eq(0).sum())))
            previous=indices
        self.assertTrue(torch.equal(result[1]['y_obs'],base))
        for level in levels[1:]:
            idx=result[level]['corrupted_indices']
            self.assertTrue(torch.equal(result[level]['y_obs'][idx],result[.8]['y_obs'][idx]))

    def test_fast_loss_and_gradient_match_reference(self):
        for dtype,tolerance in [(torch.float64,1e-10),(torch.float32,2e-6)]:
            for mu in (0.,.5,1.):
                torch.manual_seed(42)
                x=torch.randn(11,7,dtype=dtype,requires_grad=True)
                prior=torch.randn(11,7,dtype=dtype).softmax(-1)
                s=torch.tensor([0,1,1,0,1,1,1,0,1,1,1])
                old=auc_noise_only_a11_loss(x,prior,s,alpha=mu)
                old_grad=torch.autograd.grad(old,x)[0]
                new=soft_pairwise_auc(x,prior,s,mu=mu,chunk_size=3)
                new_grad=torch.autograd.grad(new,x)[0]
                torch.testing.assert_close(new,old,atol=tolerance,rtol=tolerance)
                torch.testing.assert_close(new_grad,old_grad,atol=tolerance,rtol=tolerance)

    def test_corrected_ce_uses_actual_human_labels(self):
        x=torch.tensor([[2.,-1.],[-1.,2.]],requires_grad=True)
        p=torch.full((2,2),.5)
        s=torch.zeros(2,dtype=torch.long)
        config={'mu':.5,'tau':.5,'auc_chunk_size':4,'lam':1.}
        clean=objective(x,p,torch.tensor([0,1]),s,config)[0]
        noisy=objective(x,p,torch.tensor([1,0]),s,config)[0]
        self.assertGreater(noisy.item(),clean.item())
        self.assertNotEqual(torch.autograd.grad(clean,x,retain_graph=True)[0].tolist(),torch.autograd.grad(noisy,x)[0].tolist())

    def test_no_pair_case_has_zero_gradient(self):
        x=torch.randn(3,2,requires_grad=True)
        value=soft_pairwise_auc(x,torch.full_like(x,.5),torch.tensor([0,0,1]))
        self.assertEqual(value.item(),0)
        self.assertEqual(torch.autograd.grad(value,x)[0].abs().sum().item(),0)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_compiled_cuda_loss_gradient(self):
        for n,c in [(23,7),(31,10),(19,47)]:
            torch.manual_seed(42)
            x=torch.randn(n,c,device='cuda',requires_grad=True)
            prior=torch.randn(n,c,device='cuda').softmax(-1)
            s=(torch.arange(n,device='cuda')%3!=0).long()
            old=auc_noise_only_a11_loss(x,prior,s,alpha=.5)
            old_g=torch.autograd.grad(old,x)[0]
            new=soft_pairwise_auc(x,prior,s,mu=.5,chunk_size=256)
            new_g=torch.autograd.grad(new,x)[0]
            torch.testing.assert_close(new,old,atol=2e-6,rtol=2e-5)
            torch.testing.assert_close(new_g,old_g,atol=2e-6,rtol=2e-5)


if __name__=='__main__':unittest.main()
