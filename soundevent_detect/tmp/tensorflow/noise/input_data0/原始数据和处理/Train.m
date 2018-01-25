magic_number = 3331
data_number = 10000
line_number = 50
column_number = 1
magic_number1 = 2049
data_number1 = 10000
MAG = [magic_number;data_number;line_number;column_number]
MBG = [magic_number1;data_number1]
fid=fopen('C:\Users\Administrator\Desktop\train-images.idx3-ubyte','wb')
fid1=fopen('C:\Users\Administrator\Desktop\train-labels.idx1-ubyte','wb')
fwrite(fid,MAG,'int32','b')
fwrite(fid1,MBG,'int32','b')
B = load('C:\Users\Administrator\Desktop\max_1.mat')
D = load('C:\Users\Administrator\Desktop\max_0.mat')
F = load('C:\Users\Administrator\Desktop\Label.mat')
for i = 1:10000
    
    A = rand(1,1)
    if A > 0.5
        a = 1 + 50*(i-1)
        b = 50*i
        C = B.max_1(a:b)
        G = F.Channel_1_Data(1)
        fwrite(fid,C,'float32','b')
        fwrite(fid1,G,'uint8','b')
    else
        a = 1 + 50*(i-1)
        b = 50*i
        E = D.max(a:b)
        H = F.Channel_1_Data(2)
        fwrite(fid,E,'float32','b')
        fwrite(fid1,H,'uint8','b')
    end
end
