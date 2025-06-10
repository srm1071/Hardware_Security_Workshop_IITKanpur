module main_tb;
reg A,B,C;
wire D,E;

main m(A , B , C ,D ,E);
initial
	begin
	$dumpfile("dumpfile.vcd");
	$dumpvars(0,main_tb);
	A=0;
	B=0;
	C=0;
	#100
	A=1;
	B=1;
	C=1;
	end
	initial #200
	$finish;
endmodule