% FUNCTION success = mssendraw(sock,data,len)
%
% Author: 
%   Steven Michael (smichael@ll.mit.edu)
%
% Description:
%
%    Send raw byte data "data"  over socket "sock"
%
%    "sock" is a socket handle previously created with 
%    "msaccept" or "msconnect"
%
%    "data" is a MATLAB numeric data type. "mssendraw" will send the
%       raw data as it is stored in MATLAB (Real part only). For
%       instance, if "data" is an array of size (4,2) doubles, 
%       MATLAB will send 4*2*8 bytes (double = 8 bytes) of the raw
%       binary data for this array.
%
%    "len" is an optional variable indicating the number of bytes to 
%       send.  If not specified, the entire (real part) of "data"
%       will be sent.
%
%    "success" is a status indicator. "success" < 0 indicates failure.
%      Any other number indicates success.
%
%
% Example:
%
%   sock = msconnect('hostname',port)
%   success = mssendraw(sock,rand(100));
%
