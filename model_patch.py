class DynamicAttention(nn.Module):
    def __init__(self):
        super(DynamicAttention,self).__init__()
        self.gamma=nn.Parameter(torch.zeros(1))
        self.softmax=nn.Softmax(dim=1)
    def forward(self,x,conv):
        b,_,h,w=x.size()

        Mb=torch.zeros(b,h*w,h*w).cuda()
        for _ in range(3):
            y = conv(x)#b,c,h,w


            y1=y.view(b,-1,h*w).permute(0,2,1)#b,h*w,c
            y2=y.view(b,-1,h*w)#b,c,h*w
            y3=torch.bmm(y1,y2)#b,h*w,h*w

            Mb=Mb+y3
            Mc=self.softmax(Mb)#b,N,N

            x1=x.view(b,-1,h*w)#b,c,N
            add=torch.bmm(x1,Mc)#b,c,N
            add=add.view(b,-1,h,w)
            x=x+self.gamma*add
        return x




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5,padding=2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5,padding=2)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(980, 50)
        self.fc2 = nn.Linear(50, 10)

        # self.softmax=nn.Softmax(dim=1)
        # self.gamma1=nn.Parameter(torch.zeros(1))
        # self.gamma2=nn.Parameter(torch.zeros(1))
        self.dynamicAtt1=DynamicAttention()
        self.dynamicAtt2=DynamicAttention()

    def forward(self, x):


        # del y,y1,y2,y3,Mb,Mc,x1,add
        x=self.dynamicAtt1(x,self.conv1)

        x=self.conv1(x)

        x = F.relu(F.max_pool2d(x, 2))

        x=self.dynamicAtt2(x,self.conv2)

        x = self.conv2(x)

        x = F.relu(F.max_pool2d(self.conv2_drop(x), 2))
        # print(x.size())
        x = x.view(-1, 980)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)