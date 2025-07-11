************************************************************************
file with basedata            : md361_.bas
initial value random generator: 1559567798
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  22
horizon                       :  171
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     20      0       21        9       21
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           7  17  18
   3        3          2           5  14
   4        3          2           6  11
   5        3          3           8   9  15
   6        3          2          10  21
   7        3          3           9  11  14
   8        3          3          17  18  19
   9        3          2          16  21
  10        3          2          12  18
  11        3          3          15  16  20
  12        3          3          13  15  20
  13        3          1          14
  14        3          1          16
  15        3          1          19
  16        3          1          19
  17        3          1          21
  18        3          1          20
  19        3          1          22
  20        3          1          22
  21        3          1          22
  22        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     6       0    7    5    8
         2     7       0    6    4    4
         3     9       0    5    4    3
  3      1     1       0    6    9    4
         2     3       0    4    9    2
         3     7       7    0    8    2
  4      1     1       4    0   10    4
         2     2       1    0    7    4
         3     4       0    1    3    4
  5      1     2       0    4   10    4
         2     8       6    0    9    3
         3     9       5    0    9    1
  6      1     2       9    0    4    8
         2     3       4    0    4    7
         3     4       4    0    3    6
  7      1     2       1    0    7    9
         2     4       0    9    7    9
         3    10       1    0    7    8
  8      1     2       0    3    8    8
         2     4       0    2    5    7
         3     5       0    2    3    6
  9      1     3       0    9    8    5
         2     3      10    0    7    6
         3    10       9    0    6    2
 10      1     3       0    7    3    6
         2     3       7    0    2    6
         3     9       6    0    1    5
 11      1     5       6    0    4    8
         2     7       0    4    4    4
         3    10       4    0    4    2
 12      1     4       7    0    1    8
         2     4       0    5    1    7
         3    10       7    0    1    6
 13      1     6       0    6    9    2
         2     8       0    4    6    2
         3    10       4    0    4    1
 14      1     2       8    0    6    7
         2     8       0    7    4    7
         3    10       0    5    3    6
 15      1     2       9    0    4    6
         2    10       0    6    4    1
         3    10       0    7    3    1
 16      1     1       4    0    6    5
         2     6       0    4    4    4
         3     9       0    4    3    2
 17      1     1       0    6    6    6
         2     4       0    6    4    6
         3     9       3    0    2    4
 18      1     2       1    0   10    6
         2     8       1    0    7    4
         3    10       0    8    5    3
 19      1     2       0    6    8    4
         2     6       4    0    8    3
         3     6       0    5    8    4
 20      1     2       0    3    8    9
         2     8       4    0    5    9
         3    10       3    0    4    9
 21      1     2       0    6    9    7
         2     4       0    6    8    7
         3    10       0    6    7    7
 22      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
    9   11  112  103
************************************************************************
DEADLINES:
jobnr.  deadline
  2       144
  3       29
  5       88
  6       81
  7       114
  8       8
  9       142
  10      129
  12      100
  13      32
  14      132
  15      160
  16      122
  17      156
  18      99
  19      116
  20      143
  21      128
************************************************************************
