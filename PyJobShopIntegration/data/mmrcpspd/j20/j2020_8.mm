************************************************************************
file with basedata            : md340_.bas
initial value random generator: 448557781
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  22
horizon                       :  147
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     20      0       17       12       17
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3          13  15  16
   3        3          3           5   6   7
   4        3          3          14  17  19
   5        3          2          13  14
   6        3          2          12  17
   7        3          3           8   9  20
   8        3          2          12  14
   9        3          1          10
  10        3          3          11  15  16
  11        3          2          12  18
  12        3          1          19
  13        3          1          20
  14        3          2          15  18
  15        3          1          21
  16        3          2          19  21
  17        3          2          18  20
  18        3          1          21
  19        3          1          22
  20        3          1          22
  21        3          1          22
  22        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     4       0    6    7    0
         2     5       0    4    4    0
         3    10       8    0    0    5
  3      1     2       0    4    0    8
         2     3       5    0    0    7
         3     8       0    3    8    0
  4      1     4       0    4    0   10
         2     4       7    0    0    9
         3     5       6    0    0    8
  5      1     4       0    7    5    0
         2     8       0    6    0    7
         3    10       3    0    5    0
  6      1     3       9    0    4    0
         2     5       7    0    4    0
         3     8       0    7    0    2
  7      1     2       0    6    8    0
         2     5       9    0    7    0
         3     7       7    0    0    8
  8      1     4       5    0    7    0
         2     5       0    4    3    0
         3    10       3    0    0    8
  9      1     2       3    0    4    0
         2     2       4    0    0    9
         3     2       6    0    2    0
 10      1     2       5    0    4    0
         2     4       4    0    0    5
         3     5       2    0    1    0
 11      1     2       2    0    0    3
         2     4       0    8    0    2
         3     7       2    0    6    0
 12      1     5       6    0    0    9
         2     6       0    8    0    9
         3     8       0    5   10    0
 13      1     3       0    9    6    0
         2     4       5    0    0    4
         3     5       0    4    0    4
 14      1     4       0    4    7    0
         2     8       9    0    0    7
         3    10       7    0    2    0
 15      1     2       7    0    0    8
         2     4       5    0    8    0
         3     7       0    8    0    7
 16      1     2       6    0    0    8
         2     5       0    8    5    0
         3     7       3    0    0    7
 17      1     6       7    0    7    0
         2     9       7    0    0    4
         3    10       0    3    5    0
 18      1     1       0    3   10    0
         2     5       5    0    6    0
         3     6       0    3    2    0
 19      1     1      10    0    0    6
         2     6       0    6   10    0
         3     7       5    0    0    4
 20      1     3       0    3    4    0
         2     5       4    0    2    0
         3     7       0    3    0    5
 21      1     3       0    8    0    7
         2     6       8    0    0    6
         3     8       4    0    7    0
 22      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   36   36   96   94
************************************************************************
DEADLINES:
jobnr.  deadline
  2       62
  3       99
  7       105
  9       14
  14      4
  15      104
  18      98
  21      125
************************************************************************
