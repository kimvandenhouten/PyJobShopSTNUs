************************************************************************
file with basedata            : md343_.bas
initial value random generator: 1527274423
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  22
horizon                       :  162
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     20      0       22        9       22
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3          12  13  14
   3        3          3           5   9  11
   4        3          3           7   9  20
   5        3          3           6  10  14
   6        3          1          17
   7        3          2           8  16
   8        3          2          11  21
   9        3          2          13  15
  10        3          2          12  15
  11        3          2          12  17
  12        3          1          19
  13        3          2          16  21
  14        3          2          19  20
  15        3          3          16  17  21
  16        3          1          18
  17        3          1          18
  18        3          1          19
  19        3          1          22
  20        3          1          22
  21        3          1          22
  22        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     3       7    4    7    0
         2     4       7    4    0    2
         3     8       6    3    5    0
  3      1     2      10    9    0    8
         2     6      10    7    0    5
         3    10       9    6    2    0
  4      1     2       8    7    0    4
         2     4       7    5    0    3
         3    10       6    2    0    3
  5      1     1       3    6    0    7
         2     8       3    1    6    0
         3     8       2    3    4    0
  6      1     5       3    8    0   10
         2     7       2    7    0   10
         3     8       1    6    0    9
  7      1     6       6    4    0    6
         2     8       3    4    0    5
         3     9       3    3    0    4
  8      1     3       6    8    9    0
         2     5       5    7    0    6
         3     8       5    5    7    0
  9      1     4       6    9    9    0
         2     5       4    5    8    0
         3     8       3    3    6    0
 10      1     5       8    8    4    0
         2     5       6    7    0    6
         3     6       5    6    0    4
 11      1     2       6    5    3    0
         2     4       3    4    3    0
         3    10       2    4    0    4
 12      1     2       6    5    9    0
         2     4       5    5    6    0
         3     8       5    3    0    4
 13      1     1       4    8    8    0
         2     8       2    6    7    0
         3    10       2    4    5    0
 14      1     1       3   10    6    0
         2     5       2   10    0    7
         3     5       3   10    5    0
 15      1     5       7    8    0    5
         2     7       6    7    2    0
         3    10       5    6    1    0
 16      1     2       7    6    0    7
         2     4       6    4    4    0
         3     6       5    3    3    0
 17      1     3       4    7    8    0
         2     7       4    7    7    0
         3    10       3    3    0    1
 18      1     5       7    5    0    4
         2     5       9    4    7    0
         3     7       6    3    5    0
 19      1     1      10    8   10    0
         2     4       6    6    6    0
         3     4       4    7    8    0
 20      1     6       5    7    0    6
         2     7       4    7    5    0
         3     8       3    5    4    0
 21      1     2       9    7    7    0
         2     4       8    6    0    3
         3     9       6    5    5    0
 22      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   23   32   84   72
************************************************************************
DEADLINES:
jobnr.  deadline
  2       125
  3       130
  4       66
  5       65
  6       101
  7       71
  8       43
  9       74
  10      25
  11      98
  12      24
  13      90
  14      17
  15      52
  16      43
  17      124
  18      89
  19      90
  20      8
  21      105
************************************************************************
