magic_number = 3331
data_number = 20000
line_number = 50
column_number = 1
magic_number1 = 2049
data_number1 = 20000
MAG = [magic_number;data_number;line_number;column_number]
MBG = [magic_number1;data_number1]
fid=fopen('C:\Users\Administrator.5846LFER4U1HAL1\Desktop\train-images.idx3-ubyte','wb')
fid1=fopen('C:\Users\Administrator.5846LFER4U1HAL1\Desktop\train-labels.idx1-ubyte','wb')
fwrite(fid,MAG,'int32','b')
fwrite(fid1,MBG,'int32','b')
for i = 3:12
    name = ['C:\Users\Administrator.5846LFER4U1HAL1\Desktop\normal\关盖脱水噪声跟着生产线走-',num2str(i),'.mat']
    name1 = ['C:\Users\Administrator.5846LFER4U1HAL1\Desktop\wrong\关盖脱水电机异常声-',num2str(i),'.mat']
    B = load(name)
    D = load(name1)
    F = load('C:\Users\Administrator.5846LFER4U1HAL1\Desktop\标签.mat')
    for i = 1:2000

        A = rand(1,1)
        if A > 0.5
            a = 1 + 50*(i-1)
            b = 50*i
            C = B.Channel_1_Data(a:b)
            G = F.Channel_1_Data(1)
            fwrite(fid,C,'float32','b')
            fwrite(fid1,G,'uint8','b')
        else
            a = 1 + 50*(i-1)
            b = 50*i
            E = D.Channel_1_Data(a:b)
            H = F.Channel_1_Data(2)
            fwrite(fid,E,'float32','b')
            fwrite(fid1,H,'uint8','b')
        end
    end
end
gzip('C:\Users\Administrator.5846LFER4U1HAL1\Desktop\train-images.idx3-ubyte')
gzip('C:\Users\Administrator.5846LFER4U1HAL1\Desktop\train-labels.idx1-ubyte')