%% for output of detector part
% hits
fid = fopen("HitsID.dat","rb");
tmp = fread(fid,'int32');
fclose(fid);
gpuhit = reshape(tmp,5,[]);
gpuhit = gpuhit';

fid = fopen("Hits.dat","rb");
tmp = fread(fid,'float32');
fclose(fid);
tmp = reshape(tmp,5,[]);
gpuhit = [gpuhit,tmp']; 
% now the order of columns is particle id, panel id, module id, crystal id and scatterng order
% where scattering order 1 and 2 means Compton, 3 Rayleigh and 4
% Photonelectric,deposited energy, time, local x y z

% singles
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
% the order of columns is particle id, panel id, module id, crystal id,
% site id, event id, time, deposited energy and local x y z

%% for results of PSF file
% should have three files about positions and momentums, ids and time
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
% the oreder of columns is global x y z vx vy vz, kinetic energy, particle
% id and time
