for %%A in (05 10 15 20) do  for %%Z in (05 10 15 20) do python DP_pcglns.py -i=../../pcglns/e3x_1.pcglns -UB=1179 -w=4 -b %%A -t %%Z >../../logs/dp/e3x_1.%%A-%%Z.log.txt