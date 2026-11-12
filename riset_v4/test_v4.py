import unittest
import numpy as np
from night_tracking import Half
from .features import pair,velocity
from .labels import boundary_targets,sample_weights,EDGE_SWAP,CLASS_SWAP
from .decode import decode,Settings
from .data import split,validate_split
from .models import Heads,calibrate
from .evaluate import score


def fixture(n=800):
    rng=np.random.default_rng(7)
    y=(np.arange(n)//100%3).astype(np.int8)
    ball=np.column_stack([30+np.arange(n)*.01,np.full(n,30)]).astype(np.float32)
    teams=[]
    for side in (0,1):
        p=np.repeat(ball[:,None,:],3,axis=1)+rng.normal(0,.1,(n,3,2))
        p[:,:,0]+=np.where(y==side,1.,12.)[:,None]
        teams.append(p.astype(np.float32))
    ball[y==2]=np.nan
    return Half(ball,tuple(teams),np.array([105.,68.])),y


class Tests(unittest.TestCase):
    def test_video_recovery_and_rejection(self):
        import cv2
        from .video import recover_gap
        n=12;pitch=np.array([105.,68.]);frames=[];matrices=[];truth=[]
        for t in range(n):
            frame=np.zeros((200,300),np.uint8)
            point=np.array([90.+t*1.2,100.])
            cv2.circle(frame,(round(point[0]),100),4,240,-1)
            cv2.circle(frame,(round(point[0])-1,99),1,100,-1)
            frames.append(frame);truth.append(point/[300.,200.]*pitch)
            matrices.append(np.diag([300.,200.,1.]))
        h=np.asarray(matrices);v=np.ones(n,bool);err=np.zeros(n)
        recovered,confidence,reason=recover_gap(frames,h,v,err,truth[0],truth[-1],pitch)
        self.assertEqual(reason,'accepted');self.assertLess(np.max(np.linalg.norm(recovered-truth,axis=1)),.6)
        broken=frames.copy();broken[5]=np.zeros_like(frames[5])
        self.assertIsNone(recover_gap(broken,h,v,err,truth[0],truth[-1],pitch)[0])
        v[5]=False
        self.assertEqual(recover_gap(frames,h,v,err,truth[0],truth[-1],pitch)[2],'homography')

    def test_team_symmetry_missing_and_slot_jump(self):
        d,y=fixture();(a,b),names=pair(d)
        (swapped,_),_=pair(Half(d.ball,d.players[::-1],d.pitch))
        np.testing.assert_allclose(b,swapped,equal_nan=True)
        self.assertEqual(a.shape,(len(y),len(names)))
        self.assertFalse(np.isinf(a).any())
        p=np.zeros((20,2),np.float32);p[8]=np.nan;p[15:]=100
        v=velocity(p,3,12)
        self.assertTrue(np.isnan(v[8:12]).all())
        self.assertTrue(np.isnan(v[15:18]).all())

    def test_boundaries_and_weights(self):
        y=np.array([0]*100+[1]*10+[0]*100+[3]*5+[2]*50)
        target=boundary_targets(y)
        self.assertEqual(target[100,0],1)
        self.assertEqual(target[110,1],1)
        self.assertEqual(target[215].sum(),0)
        w=sample_weights(y)
        self.assertGreater(w[105],w[50]);self.assertEqual(w[211],0)
        mirror=boundary_targets(CLASS_SWAP[y])
        np.testing.assert_equal(target[:,EDGE_SWAP],mirror)

    def test_decoder_can_create_missing_short_spell(self):
        p=np.tile([.8,.15,.05],(150,1));p[70:78]=[.35,.6,.05]
        q=np.zeros((150,6));q[70,0]=.9;q[78,1]=.9
        old=decode(p);new=decode(p,q,Settings())
        self.assertFalse((old==1).any())
        self.assertTrue((new[70:78]==1).all())
        np.testing.assert_equal(decode(p[:,[1,0,2]],q[:,EDGE_SWAP]),CLASS_SWAP[new])

    def test_official_metric_and_disjoint_splits(self):
        y=np.array([0]*100+[1]*50+[2]*10+[0]*30,dtype=np.int8)
        out=score([y],[y]);self.assertAlmostEqual(out['spellF1'],1.)
        for f in range(5):
            tr,cal,te=split(f);self.assertEqual(len(tr),3)
        with self.assertRaises(ValueError):validate_split(['J03WMX'],['J03WMX'],['J03WN1'])

    def test_train_predict_calibrate(self):
        d,y=fixture(2000);features,_=pair(d)
        model=Heads().fit([(features,y,1.)],iterations=3)
        p,q=model.raw(features)
        cal=calibrate([p],[q],[y]);cp,cq=cal.state(p),cal.edge(q)
        self.assertEqual(cq.shape,(2000,6));self.assertTrue(np.isfinite(cp).all())
        np.testing.assert_allclose(cp.sum(1),1.,atol=1e-6)
        np.testing.assert_allclose(cal.edge(q[:,EDGE_SWAP]),cq[:,EDGE_SWAP],atol=1e-6)

    def test_tcn_chunk_equivalence_and_training(self):
        import torch
        from .tcn import Network,Temporal
        torch.manual_seed(3);torch.set_num_threads(2)
        net=Network(8,channels=8).eval();x=torch.randn(1,8,1300)
        with torch.inference_mode():
            full=net(x)
            for start,end in [(0,400),(400,800),(800,1300)]:
                a=max(0,start-net.halo);b=min(1300,end+net.halo)
                chunk=net(x[:,:,a:b])
                for s,t in zip(full,chunk):
                    torch.testing.assert_close(s[:,:,start:end],t[:,:,start-a:end-a],atol=2e-5,rtol=2e-5)
        d,y=fixture(900);features,_=pair(d)
        temporal=Temporal().fit([(features,y,1.)],epochs=1,steps=1,device='cpu',progress=lambda x:None)
        p,q=temporal.raw(features,chunk=300)
        self.assertEqual(p.shape,(900,3));self.assertTrue(np.isfinite(q).all())


if __name__=='__main__':unittest.main()
