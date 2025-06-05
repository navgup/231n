

class ChartClfDataset(Dataset):
    def __init__(self,csv:str):
        df=pd.read_csv(csv); self.paths=df['path'].tolist(); self.labels=df['label'].tolist()
        self.tf=transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3,[0.5]*3),
        ])
    def __len__(self): return len(self.paths)
    def __getitem__(self,idx):
        img=Image.open(self.paths[idx]).convert("RGB")
        return self.tf(img), torch.tensor(self.labels[idx],dtype=torch.long)

class ViTClassifier(nn.Module):
    def __init__(self,num_classes:int=3,model="vit_tiny_patch16_224"):
        super().__init__(); self.backbone=timm.create_model(model,pretrained=False,num_classes=0); self.head=nn.Linear(self.backbone.embed_dim,num_classes)
    def forward(self,x): return self.head(self.backbone(x))

def train_classifier(csv_path:str,epochs:int=20,batch:int=64,lr:float=3e-4,device="cuda"):
    ds=ChartClfDataset(csv_path); split=int(len(ds)*0.8)
    tr,val=torch.utils.data.random_split(ds,[split,len(ds)-split],generator=torch.Generator().manual_seed(42))
    dl_tr=DataLoader(tr,batch_size=batch,shuffle=True,num_workers=4,pin_memory=True)
    dl_val=DataLoader(val,batch_size=batch,shuffle=False,num_workers=4)
    model=ViTClassifier().to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4); ce=nn.CrossEntropyLoss()
    for ep in range(1,epochs+1):
        model.train(); tl=0
        for x,y in dl_tr:
            x,y=x.to(device),y.to(device); opt.zero_grad(); loss=ce(model(x),y); loss.backward(); opt.step(); tl+=loss.item()
        model.eval(); vl=0; correct=tot=0
        with torch.no_grad():
            for x,y in dl_val:
                x,y=x.to(device),y.to(device); logits=model(x); vl+=ce(logits,y).item(); pred=logits.argmax(1); correct+=(pred==y).sum().item(); tot+=y.size(0)
        print(f"Ep{ep:03d}|Train{tl/len(dl_tr):.4f}|Val{vl/len(dl_val):.4f}|Acc{correct/tot:.3f}")
    torch.save(model.state_dict(), os.path.join(os.path.dirname(csv_path),"vit_3class_224.pth"))
