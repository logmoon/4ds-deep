import cv2, numpy as np, base64, io, os, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image

BLUR_THRESHOLD=100.0; DARK_THRESHOLD=50.0; BRIGHT_THRESHOLD=220.0
CONTRAST_THRESHOLD=40.0; RESOLUTION_MIN=100; QUALITY_THRESHOLD=0.55
DEVICE=torch.device("cpu"); IMG_SIZE=256
ENHANCER_PATH=os.path.join(os.path.dirname(__file__),"enhancer_paper (2).pth")
_enhancer=None

class RecursiveBlock(nn.Module):
    def __init__(self,ch,recursions=2):
        super().__init__(); self.recursions=recursions
        self.conv=nn.Sequential(nn.Conv2d(ch,ch,3,padding=1),nn.BatchNorm2d(ch),nn.ReLU(inplace=True),nn.Conv2d(ch,ch,3,padding=1),nn.BatchNorm2d(ch))
        self.relu=nn.ReLU(inplace=True)
    def forward(self,x):
        out=x
        for _ in range(self.recursions): out=self.relu(out+self.conv(out))
        return out

class RecursiveAutoencoder(nn.Module):
    def __init__(self,recursions=2):
        super().__init__()
        self.enc_in=nn.Sequential(nn.Conv2d(3,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(inplace=True))
        self.enc1=RecursiveBlock(64,recursions); self.pool1=nn.MaxPool2d(2)
        self.enc2_in=nn.Sequential(nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(inplace=True))
        self.enc2=RecursiveBlock(128,recursions); self.pool2=nn.MaxPool2d(2)
        self.enc3_in=nn.Sequential(nn.Conv2d(128,256,3,padding=1),nn.BatchNorm2d(256),nn.ReLU(inplace=True))
        self.enc3=RecursiveBlock(256,recursions); self.pool3=nn.MaxPool2d(2)
        self.bottleneck=nn.Sequential(nn.Conv2d(256,256,3,padding=1),nn.BatchNorm2d(256),nn.ReLU(inplace=True),RecursiveBlock(256,recursions),nn.Conv2d(256,256,3,padding=1),nn.BatchNorm2d(256),nn.ReLU(inplace=True))
        self.up3=nn.Upsample(scale_factor=2,mode="bilinear",align_corners=True)
        self.dec3_in=nn.Sequential(nn.Conv2d(512,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(inplace=True))
        self.dec3=RecursiveBlock(128,recursions)
        self.up2=nn.Upsample(scale_factor=2,mode="bilinear",align_corners=True)
        self.dec2_in=nn.Sequential(nn.Conv2d(256,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(inplace=True))
        self.dec2=RecursiveBlock(64,recursions)
        self.up1=nn.Upsample(scale_factor=2,mode="bilinear",align_corners=True)
        self.dec1_in=nn.Sequential(nn.Conv2d(128,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(inplace=True))
        self.dec1=RecursiveBlock(32,recursions)
        self.out=nn.Sequential(nn.Conv2d(32,16,3,padding=1),nn.ReLU(inplace=True),nn.Conv2d(16,3,3,padding=1),nn.Sigmoid())
    def forward(self,x):
        e1=self.enc1(self.enc_in(x)); e2=self.enc2(self.enc2_in(self.pool1(e1)))
        e3=self.enc3(self.enc3_in(self.pool2(e2))); bn=self.bottleneck(self.pool3(e3))
        d3=self.dec3(self.dec3_in(torch.cat([self.up3(bn),e3],dim=1)))
        d2=self.dec2(self.dec2_in(torch.cat([self.up2(d3),e2],dim=1)))
        d1=self.dec1(self.dec1_in(torch.cat([self.up1(d2),e1],dim=1)))
        return self.out(d1)

def _load_enhancer():
    global _enhancer
    if _enhancer is not None: return True
    if not os.path.exists(ENHANCER_PATH): return False
    m=RecursiveAutoencoder(recursions=2).to(DEVICE)
    ckpt=torch.load(ENHANCER_PATH,map_location=DEVICE)
    m.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    m.eval(); _enhancer=m; print("✅ Recursive Autoencoder loaded"); return True

def enhance_image(img_pil):
    if not _load_enhancer(): return img_pil
    orig=img_pil.size
    resized=img_pil.resize((IMG_SIZE,IMG_SIZE),Image.LANCZOS)
    t=torch.from_numpy(np.array(resized).astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(DEVICE)
    with torch.no_grad(): out=_enhancer(t)
    arr=(out.squeeze(0).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
    return Image.fromarray(arr).resize(orig,Image.LANCZOS)

def _load_image(image_path=None,image_b64=None):
    if image_path: return cv2.imread(image_path)
    if image_b64:
        try: arr=np.frombuffer(base64.b64decode(image_b64),np.uint8); return cv2.imdecode(arr,cv2.IMREAD_COLOR)
        except: return None
    return None

def _normalize(v,lo,hi): return float(np.clip((v-lo)/(hi-lo),0.0,1.0))
def _pil_to_b64(img):
    buf=io.BytesIO(); img.save(buf,format="JPEG",quality=92)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def _check_quality(img_cv2):
    gray=cv2.cvtColor(img_cv2,cv2.COLOR_BGR2GRAY); issues=[]
    h,w=gray.shape
    if w<RESOLUTION_MIN or h<RESOLUTION_MIN: issues.append("Image resolution is too low")
    br=cv2.Laplacian(gray,cv2.CV_64F).var(); sh=round(_normalize(br,0,500),2)
    if br<BLUR_THRESHOLD: issues.append("Image appears blurry or out of focus")
    bv=gray.mean(); bri=round(_normalize(bv,0,255),2)
    if bv<DARK_THRESHOLD: issues.append("Image is too dark")
    elif bv>BRIGHT_THRESHOLD: issues.append("Image is overexposed")
    cv_val=gray.std(); con=round(_normalize(cv_val,0,128),2)
    if cv_val<CONTRAST_THRESHOLD: issues.append("Low contrast detected")
    score=round(0.5*sh+0.3*bri+0.2*con,2); ok=score>=QUALITY_THRESHOLD
    return ok,score,{"sharpness":sh,"brightness":bri,"contrast":con},issues

def analyze_image(image_path=None,image_b64=None):
    img=_load_image(image_path,image_b64)
    if img is None:
        return {"ok":False,"quality_score":0.0,"details":{"sharpness":0.0,"brightness":0.0,"contrast":0.0},"issues":["Could not read image"],"reason":"Image quality does not meet publication standards"}
    ok,score,details,issues=_check_quality(img)
    result={"ok":ok,"quality_score":score,"details":details,"issues":issues}
    if not ok: result["reason"]="Image quality does not meet publication standards"
    if not ok and issues and _load_enhancer():
        img_pil=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        enh_pil=enhance_image(img_pil)
        enh_cv=cv2.cvtColor(np.array(enh_pil),cv2.COLOR_RGB2BGR)
        e_ok,e_score,e_det,_=_check_quality(enh_cv); e_det["quality_score"]=e_score
        result["enhanced_image"]=_pil_to_b64(enh_pil); result["enhanced_ok"]=e_ok; result["enhanced_details"]=e_det
    return result