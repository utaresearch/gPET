% for output of detector part
filename = "xxx.dat";
fid = fopen(filename,"rb");
start=ftell(fid);
fseek(fid,0,'eof');
stop=ftell(fid);
num = (stop-start)/48;
gpusin = zeros(num,11);
frewind(fid);
for i = 1:num
    gpusin(i,1:6) = fread(fid,6,'int32');
    gpusin(i,7) = fread(fid,1,'float64');
    gpusin(i,8:11) = fread(fid,4,'float32');
end
fclose(fid);

% for results of PSF file
% should have three files about positions, ids and time
fid = fopen("outsource.dat","rb");
tmp = fread(fid,'float32');
fclose(fid);
psf = reshape(tmp,7,[]);
psf = psf';

fid = fopen("idsource.dat.dat","rb");
tmp = fread(fid,'int32');
fclose(fid);
psf = [psf,tmp];

fid = fopen("timesource.dat","rb");
tmp = fread(fid,'float64');
fclose(fid);
psf = [psf,tmp];

