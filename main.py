import tools
import data_load
import argparse
from models import *
import tools
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformer import transform_train, transform_test,transform_target
from torch.optim.lr_scheduler import MultiStepLR


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, help='initial learning rate', default=0.01)
parser.add_argument('--save_dir', type=str, help='dir to save model files', default='saves')
parser.add_argument('--dataset', type = str, help = 'mnist, cifar10, or cifar100', default = 'cifar10')
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--noise_type', type=str, default='asymmetric') #symmetric  #flip
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.2)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--device', type=int, default=1)
parser.add_argument('--weight_decay', type=float, help='weight_decay for training', default=1e-4)
parser.add_argument('--lam', type = float, default =0.3)
parser.add_argument('--anchor', action='store_false')
parser.add_argument('--n_epoch', type=int, default=60)
parser.add_argument('--warmup_epoch', type=int, default=10)
parser.add_argument('--nums', type=int, default=0)

args = parser.parse_args()
np.set_printoptions(precision=2,suppress=True)

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
# GPU
device = torch.device('cuda:'+ str(args.device))

loss_func_ce = F.nll_loss


#est_T = t.detach().cpu().numpy()
#estimate_error = tools.error(est_T, train_data.t)

def adjust_learning_rate(n=0.1):
    # 自定义学习率下降规则（相当于每过opt.epochs轮, 学习率就会乘上0.1）
    # 其中opt.lr为初始学习率, opt.epochs为自定义值
    lr = args.lr * n
    return lr

def warmup(train_data, train_loader, model,optimizer_model, scheduler_model):
    model.train()
    
    train_loss = 0.

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer_model.zero_grad()

        clean = model(batch_x)

        ce_loss = loss_func_ce(clean.log(), batch_y.long())
        res = torch.mean(torch.sum(clean.log() * clean, dim=1))
        loss = ce_loss + res
        
        train_loss += loss.item()

        loss.backward()
        optimizer_model.step()


    print('Warmup Loss: {:.6f}'.format(train_loss / (len(train_data))*args.batch_size))


def train(train_data, train_loader,model,trans_for,trans_back,optimizer_es,optimizer_trans_for,optimizer_trans_back,scheduler1,scheduler2,scheduler3):
    model.train()
    trans_for.train()
    trans_back.train()

    train_loss = 0.
    train_acc = 0.

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = torch.zeros(batch_x.size(0), args.num_classes).scatter_(1, batch_y.view(-1,1), 1)
        batch_y = batch_y.to(device)

        clean = model(batch_x)
        noise = F.softmax(batch_y, 1)

        t_for = trans_for()
        t_back = trans_back()

        out = torch.mm(clean, t_for)
        out1 = torch.mm(noise, t_back)

        noise_y = torch.max(noise, dim=1)[1].detach()
        clean_y = torch.max(clean, dim=1)[1].detach()
        ce_loss = loss_func_ce(out.log(), noise_y.long())
        ce_loss_1 =  loss_func_ce(out1.log(), clean_y.long())
        
        for_back_1 = torch.mm(clean, t_for.detach())
        for_back = torch.mm(for_back_1, t_back.detach())
        
        loss_for_back = loss_func_ce(for_back.log(), clean_y.long())  #loss_func_ce(for_back.log(), clean_y.long()) #-torch.mean(torch.sum(for_back.log() * clean,dim=1)) #
        
        
        loss = ce_loss  + ce_loss_1+ args.lam * loss_for_back  

        train_loss += loss.item()
        
        pred = torch.max(out, 1)[1]
        train_correct = (pred == noise_y).sum()
        train_acc += train_correct.item()
        
        optimizer_es.zero_grad()
        optimizer_trans_for.zero_grad()
        optimizer_trans_back.zero_grad()
        
        loss.backward()
        
        optimizer_es.step()
        optimizer_trans_for.step()
        optimizer_trans_back.step()

    print('Train Loss: {:.6f},  Acc: {:.6f}'.format(train_loss / (len(train_data))*args.batch_size,  train_acc / (len(train_data))))

    scheduler1.step()
    scheduler2.step()
    scheduler3.step()


def val(val_data, val_loader, model, trans):
    val_acc = 0.
    val_loss = 0.

    with torch.no_grad():
        model.eval()
        trans.eval()
        
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            clean = model(batch_x)
            t = trans()

            out = torch.mm(clean, t)
            loss = loss_func_ce(out.log(), batch_y.long())
            val_loss += loss.item()
            pred = torch.max(out, 1)[1]
            val_correct = (pred == batch_y).sum()
            val_acc += val_correct.item()

            
    print('Val Loss: {:.6f}, Acc: {:.6f}'.format(val_loss / (len(val_data))*args.batch_size, val_acc / (len(val_data))))

def test(test_data, test_loader, model):
    eval_loss = 0.
    eval_acc = 0.
    
    with torch.no_grad():
        model.eval()
    
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            clean = model(batch_x)
    
            loss = loss_func_ce(clean.log(), batch_y.long())
            eval_loss += loss.item()
            pred = torch.max(clean, 1)[1]
            eval_correct = (pred == batch_y).sum()
            eval_acc += eval_correct.item()
    
        print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(test_data)) * args.batch_size,
                                                      eval_acc / (len(test_data))))
    return eval_acc / (len(test_data)) 

def main():
    #init
    print(args)
    #log = open('./save/T2/{}/{}_{}_{}.txt'.format(args.dataset, args.noise_type, args.noise_rate,args.nums),'w')
    
    if args.dataset == 'mnist':
    
        args.n_epoch = 60
        num_classes = 10
        milestones = None
    
        train_data = data_load.mnist_dataset(True, transform=transform_train(args.dataset), target_transform=transform_target,
                                             noise_rate=args.noise_rate, random_seed=args.seed, noise_type=args.noise_type,anchor=args.anchor)
        val_data = data_load.mnist_dataset(False, transform=transform_test(args.dataset), target_transform=transform_target,
                                           noise_rate=args.noise_rate, random_seed=args.seed, noise_type=args.noise_type)
        test_data = data_load.mnist_test_dataset(transform=transform_test(args.dataset), target_transform=transform_target)
        model = Lenet()
        trans = sig_t(device, args.num_classes)
        trans_1 = sig_t(device, args.num_classes)
        optimizer_trans = optim.Adam(trans.parameters(), lr=args.lr, weight_decay=0)
        optimizer_trans_1 = optim.SGD(trans_1.parameters(), lr=args.lr, weight_decay=0, momentum=0.9)
        
        t_true = train_data.t
        print('Translate matrix:', file=log, flush=True)
        print(t_true, file=log, flush=True)
    
    if args.dataset == 'cifar10':
    
        args.num_classes = 10
        args.n_epoch = 60
        milestones = [30,45]
    
        train_data = data_load.cifar10_dataset(True, transform=transform_train(args.dataset), target_transform=transform_target,
                                             noise_rate=args.noise_rate, random_seed=args.seed, noise_type=args.noise_type, anchor=args.anchor)
        val_data = data_load.cifar10_dataset(False, transform=transform_test(args.dataset), target_transform=transform_target,
                                           noise_rate=args.noise_rate, random_seed=args.seed, noise_type=args.noise_type)
        test_data = data_load.cifar10_test_dataset(transform=transform_test(args.dataset), target_transform=transform_target)
        model = ResNet34(args.num_classes)
        trans = sig_t(device, args.num_classes)
        trans_1 = sig_t(device, args.num_classes)
        optimizer_trans = optim.SGD(trans.parameters(), lr=args.lr, weight_decay=0, momentum=0.9)
        optimizer_trans_1 = optim.SGD(trans_1.parameters(), lr=args.lr, weight_decay=0, momentum=0.9)

    
    if args.dataset == 'cifar100':
        args.init = 4.5
    
        args.num_classes = 100
        args.n_epoch = 100
    
        milestones = [30, 60]
    
        train_data = data_load.cifar100_dataset(True, transform=transform_train(args.dataset), target_transform=transform_target,
                                             noise_rate=args.noise_rate, random_seed=args.seed, noise_type=args.noise_type, anchor=args.anchor)
        val_data = data_load.cifar100_dataset(False, transform=transform_test(args.dataset), target_transform=transform_target,
                                           noise_rate=args.noise_rate, random_seed=args.seed, noise_type=args.noise_type)
        test_data = data_load.cifar100_test_dataset(transform=transform_test(args.dataset), target_transform=transform_target)
    
        model = ResNet34(args.num_classes)
        trans = sig_t(device, args.num_classes, init=args.init)
        optimizer_trans = optim.Adam(trans.parameters(), lr=args.lr, weight_decay=0)
        trans_1 = sig_t(device, args.num_classes)
        optimizer_trans_1 = optim.SGD(trans_1.parameters(), lr=args.lr, weight_decay=0, momentum=0.9)

    
    #optimizer and StepLR
    optimizer_es = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    scheduler1 = MultiStepLR(optimizer_es, milestones=milestones, gamma=0.1)
    scheduler2 = MultiStepLR(optimizer_trans, milestones=milestones, gamma=0.1)
    scheduler3 = MultiStepLR(optimizer_trans_1, milestones=milestones, gamma=0.1)
    
    
    #data_loader
    train_loader = DataLoader(dataset=train_data, 
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=4,
                              drop_last=False)
    
    val_loader = DataLoader(dataset=val_data,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=4,
                            drop_last=False)
    
    test_loader = DataLoader(dataset=test_data,
                             batch_size=args.batch_size,
                             num_workers=4,
                             drop_last=False)
    
    #cuda
    if torch.cuda.is_available:
        model = model.to(device)
        trans = trans.to(device)
        trans_1 = trans_1.to(device)

    best_acc = 0
    best_acc_back = 0
    #warmup
    for epoch in range(args.warmup_epoch):
        print('epoch[{}], Warmup'.format(epoch + 1))
        warmup(train_data, train_loader, model,optimizer_es, scheduler1)
        val(val_data, val_loader, model, trans)
        acc = test(test_data, test_loader, model)
        if acc> best_acc:
            best_acc = acc
        print('Best_acc: {:.6f}'.format(best_acc))

    acc_list = []
    for epoch in range(args.n_epoch):
        print('epoch[{}], Train'.format(epoch+1))
        train(train_data,train_loader,model,trans,trans_1,optimizer_es,optimizer_trans,optimizer_trans_1,scheduler1,scheduler2,scheduler3)
        val(val_data, val_loader, model, trans)
        acc = test(test_data, test_loader, model)
        
        acc_list.append(acc)
        
        if acc> best_acc:
            best_acc = acc


        print('Best_acc: {:.6f}'.format(best_acc))


    print('Best_acc: ', best_acc)
    return best_acc,acc_list
        
if __name__=='__main__':
    acc,acc_list = main()
    #noise_rate_list = [0.2,0.4,0.6]
    '''noise_rate_list = [0.4]
    nums = 3
    acc_list = []
    for i in range(len(noise_rate_list)):
        args.noise_rate = noise_rate_list[i]
        acc_list.append([])
        for j in range(nums):
            args.nums = j+1
            acc = main()
            acc_list[i].append(acc)
        print('noise rate: {}'.format(args.noise_rate))
        print(acc_list[i])
        
    np.set_printoptions(precision=4)
    acc_list = np.array(acc_list)
    print('\nTest acc :')
    print(acc_list)'''
  
    '''noise_rate_list = [0.4]
    #lam_list = [0.1,0.3,0.5,0.7]
    lam_list = [0.5]
    nums = 3
    args.noise_rate = 0.2
    
    for i in lam_list:
        args.lam = i
        #log1 = open('./save/T2/{}/error_{}_{}_lam_{}.txt'.format(args.dataset, args.noise_type, args.noise_rate,args.lam),'w')
        #log2 = open('./save/T2/{}/lam_acc_{}_{}_lam_{}.txt'.format(args.dataset, args.noise_type, args.noise_rate,args.lam),'w')
        for j in range(nums):
            args.nums = j+1
            acc,error_list_epoch,acc_list = main()
            print('Acc: ', acc)
            #print(error_list_epoch , file=log1, flush=True)
            #print(acc_list , file=log2, flush=True)
        print('noise rate: {}'.format(args.noise_rate))
        print(args)
        
    np.set_printoptions(precision=6)
    acc_list = np.array(acc_list)'''
