************************************************************************
file with basedata            : md364_.bas
initial value random generator: 282472051
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  22
horizon                       :  166
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     20      0       22        2       22
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           9  11  16
   3        3          2          12  19
   4        3          3           5   6   7
   5        3          3          10  11  14
   6        3          3          12  15  21
   7        3          3           8  13  21
   8        3          2          10  20
   9        3          3          12  15  17
  10        3          1          19
  11        3          3          13  17  20
  12        3          1          18
  13        3          1          19
  14        3          2          16  18
  15        3          1          18
  16        3          1          17
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
  2      1     4       9    0    8    6
         2    10       0   10    7    6
         3    10       8    0    6    6
  3      1     7       0    9    7    6
         2    10       1    0    6    4
         3    10       0    5    5    4
  4      1     1       8    0   10    7
         2     1       7    0   10    8
         3     5       6    0   10    4
  5      1     5       0    3    5    4
         2     7       5    0    4    4
         3    10       2    0    2    3
  6      1     1       0    6    9    4
         2     6       0    5    8    4
         3    10       1    0    4    4
  7      1     4       0    7    8    6
         2     5       0    7    7    5
         3     6       0    6    5    4
  8      1     5       0    5    7    8
         2     6       8    0    4    7
         3     6       0    4    4    7
  9      1     1       0   10    8    7
         2     4       0   10    7    6
         3     9       0   10    3    5
 10      1     6       0    9    7   10
         2     7       0    4    6    1
         3     7       3    0    6    1
 11      1     1       0    4    7    8
         2     5       0    3    5    7
         3     9       4    0    1    7
 12      1     2       0    7    8    7
         2     7       7    0    8    7
         3     9       5    0    8    5
 13      1     1      10    0    5    2
         2     5       0    8    5    2
         3    10       0    7    5    1
 14      1     1       6    0    7    9
         2     4       3    0    7    7
         3     5       0    8    5    4
 15      1     5       1    0    4    8
         2     9       0    8    2    6
         3     9       1    0    3    5
 16      1     3       0   10   10    7
         2     4       0    8    9    5
         3    10       0    8    9    4
 17      1     6       0    6    9    4
         2     6       0    5    8    5
         3    10       4    0    6    3
 18      1     1       0    8    6    7
         2     5      10    0    3    5
         3     5       0    8    4    4
 19      1     6       7    0    8    8
         2     7       0    5    5    7
         3     8       6    0    3    4
 20      1     1       0    3    5    5
         2     7       6    0    4    5
         3     8       6    0    4    3
 21      1     1       0    2    4   10
         2     5       0    2    3    9
         3    10       0    2    1    8
 22      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   24   42  117  111
************************************************************************
DEADLINES:
jobnr.  deadline
  2       91
  3       159
  4       77
  5       101
  6       115
  7       31
  8       134
  9       124
  10      98
  11      68
  12      10
  13      75
  14      78
  15      110
  16      73
  17      91
  18      150
  19      44
  20      52
  21      145
************************************************************************
