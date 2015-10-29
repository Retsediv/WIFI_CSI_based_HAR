%%
%% =====================================================================================
%%       Filename:  read_log_file.m 
%%
%%    Description:  extract the CSI, payload, and packet status information from the log
%%                  file
%%        Version:  1.0
%%
%%         Author:  Yaxiong Xie 
%%         Email :  <xieyaxiongfly@gmail.com>
%%   Organization:  WANDS group @ Nanyang Technological University 
%%
%%   Copyright (c)  WANDS group @ Nanyang Technological University
%% =====================================================================================
%%

function ret = read_log_file(filename)
f = fopen(filename, 'rb');
if (f < 0)
    error('couldn''t open file %s', filename);
    return;
end

status = fseek(f, 0, 'eof');
if status ~= 0
    [msg, errno] = ferror(f);
    error('Error %d seeking: %s', errno, msg);
    fclose(f);
    return;
end
len = ftell(f);
fprintf('file length is:%d\n',len);

status = fseek(f, 0, 'bof');
if status ~= 0
    [msg, errno] = ferror(f);
    error('Error %d seeking: %s', errno, msg);
    fclose(f);
    return;
end

ret = cell(ceil(len / 420),1);
cur = 0;
count = 0;

while cur < (len - 4)
% 	field_len = fread(f, 1, 'uint16', 0, 'ieee-be');
    field_len = fread(f, 1, 'uint16', 0, 'ieee-le');
	cur = cur + 2;
    fprintf('Block length is:%d\n',field_len);

	if (cur + field_len) > len
   		break;
    end
    
%     timestamp = fread(f, 1, 'uint64', 0, 'ieee-be');
    timestamp = fread(f, 1, 'uint64', 0, 'ieee-le.l64');
	csi_matrix.timestamp = timestamp;
	cur = cur + 8;
	fprintf('timestamp is %d\n',timestamp);

% 	csi_len = fread(f, 1, 'uint16', 0, 'ieee-be');
    csi_len = fread(f, 1, 'uint16', 0, 'ieee-le');
	csi_matrix.csi_len = csi_len;
	cur = cur + 2;
    fprintf('csi_len is %d\n',csi_len);

%     tx_channel = fread(f, 1, 'uint16', 0, 'ieee-be');
    tx_channel = fread(f, 1, 'uint16', 0, 'ieee-le');
	csi_matrix.channel = tx_channel;
	cur = cur + 2;
    fprintf('channel is %d\n',tx_channel);
   
    err_info = fread(f, 1,'uint8=>int');
    csi_matrix.err_info = err_info;
    fprintf('err_info is %d\n',err_info);
    cur = cur + 1;
    
    noise_floor = fread(f, 1, 'uint8=>int');
	csi_matrix.noise_floor = noise_floor;
	cur = cur + 1;
    fprintf('noise_floor is %d\n',noise_floor);
    
    Rate = fread(f, 1, 'uint8=>int');
	csi_matrix.Rate = Rate;
	cur = cur + 1;
	fprintf('rate is %x\n',Rate);
    
%     bandWidth = fread(f, 1, 'uint8=>int');
% 	csi_matrix.bandWidth = bandWidth;
% 	cur = cur + 1;
% 	fprintf('bandWidth is %d\n',bandWidth);
    
    bandWidth = fread(f, 1, 'uint8=>int');
	csi_matrix.bandWidth = bandWidth;
	cur = cur + 1;
	fprintf('bandWidth is %d\n',bandWidth);
    
    num_tones = fread(f, 1, 'uint8=>int');
	csi_matrix.num_tones = num_tones;
	cur = cur + 1;
	fprintf('num_tones is %d  ',num_tones);

	nr = fread(f, 1, 'uint8=>int');
	csi_matrix.nr = nr;
	cur = cur + 1;
	fprintf('nr is %d  ',nr);

	nc = fread(f, 1, 'uint8=>int');
	csi_matrix.nc = nc;
	cur = cur + 1;
	fprintf('nc is %d\n',nc);
	
	rssi = fread(f, 1, 'uint8=>int');
	csi_matrix.rssi = rssi;
	cur = cur + 1;
	fprintf('rssi is %d\n',rssi);

	rssi1 = fread(f, 1, 'uint8=>int');
	csi_matrix.rssi1 = rssi1;
	cur = cur + 1;
	fprintf('rssi1 is %d\n',rssi1);

	rssi2 = fread(f, 1, 'uint8=>int');
	csi_matrix.rssi2 = rssi2;
	cur = cur + 1;
	fprintf('rssi2 is %d\n',rssi2);

	rssi3 = fread(f, 1, 'uint8=>int');
	csi_matrix.rssi3 = rssi3;
	cur = cur + 1;
	fprintf('rssi3 is %d\n',rssi3);
    
%     not_sounding = fread(f, 1, 'uint8=>int');
%     cur = cur + 1;
%     hwUpload = fread(f, 1, 'uint8=>int');
%     cur = cur + 1;
%     hwUpload_valid = fread(f, 1, 'uint8=>int');
%     cur = cur + 1;
%     hwUpload_type = fread(f, 1, 'uint8=>int');
%     cur = cur + 1;
    
    payload_len = fread(f, 1, 'uint16', 0, 'ieee-le');
	csi_matrix.payload_len = payload_len;
	cur = cur + 2;
    fprintf('payload length: %d\n',payload_len);	
    
    if csi_len > 0
        csi_buf = fread(f, csi_len, 'uint8=>uint8');
	    csi = read_csi(csi_buf, nr, nc, num_tones);
    	cur = cur + csi_len;
	    csi_matrix.csi = csi;
    else
        csi_matrix.csi = 0;
    end       
    
    if payload_len > 0
        data_buf = fread(f, payload_len, 'uint8=>uint8');	    
    	cur = cur + payload_len;
	    csi_matrix.payload = data_buf;
    else
        csi_matrix.payload = 0;
    end
    
    
    
    if (cur + 420 > len)
        break;
    end
    count = count + 1;
    ret{count} = csi_matrix;
end
if (count >1)
	ret = ret(1:(count-1));
else
	ret = ret(1);
end
fclose(f);
%end
