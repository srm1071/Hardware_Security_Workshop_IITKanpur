module main(A, B, C, D, E);
   
   output D, E;
   input  A, B, C;
   wire   w1;

   and G1(w1, A, B);
   not G2(E, C);
   or  G3(D, w1, E);

endmodule

module testbench_main();
wire D,E;
reg A,B,C;
main m(A, B, C, D, E);
initial
	begin
		A=1'b0;
		B=1'b0;
		C=1'b0;
		#100
		A=1'b1;
		B=1'b1;
		C=1'b1;
	end
		
	initial #200 $finish;
endmodule