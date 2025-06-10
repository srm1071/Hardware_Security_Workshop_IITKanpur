`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 12/09/2024 04:58:12 PM
// Design Name: 
// Module Name: montgomery_kyber
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////
module montgomery_kyber(clk,a,b,out);
input clk;
input [11:0] a,b;
output [11:0] out;
wire [23:0] res;
reg [23:0] res_t;
wire [35:0] restd;
wire [11:0] m;
wire [23:0] inter_n;
wire [24:0] resk;
wire [12:0] resu;
wire [12:0] modprime;
wire [11:0] prime;
assign prime=12'd3329;

assign res=a*b;
always@(posedge clk) begin
    res_t<=res;
end
assign restd=res_t*12'd3327;
assign m=restd[11:0];
assign inter_n=m*prime;
assign resk=res_t+inter_n;
assign resu=resk>>12;
assign modprime=resu-prime;
assign out=modprime[12]?resu[11:0]:modprime[11:0];

endmodule
