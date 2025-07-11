************************************************************************
file with basedata            : md358_.bas
initial value random generator: 741109707
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  22
horizon                       :  157
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     20      0       27       18       27
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           5   7  11
   3        3          2           6  11
   4        3          2           9  13
   5        3          3           6  12  17
   6        3          3           8  13  15
   7        3          1          10
   8        3          3           9  10  21
   9        3          1          14
  10        3          2          16  20
  11        3          2          12  17
  12        3          3          13  14  15
  13        3          2          18  21
  14        3          1          16
  15        3          3          16  20  21
  16        3          1          18
  17        3          1          20
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
  2      1     3       9    6    7    8
         2     3       6    7    8    8
         3     5       1    3    6    7
  3      1     2       8    6    7    3
         2     4       6    6    6    3
         3     8       4    5    4    3
  4      1     1       4    2    8    7
         2     4       4    1    8    6
         3     6       4    1    8    4
  5      1     1       3    6    8    7
         2     3       3    6    7    5
         3     7       2    6    5    4
  6      1     3       8    9    9    6
         2    10       5    9    6    3
         3    10       6    9    5    4
  7      1     5       9    3    9    5
         2     6       6    2    7    3
         3    10       5    2    7    3
  8      1     3      10   10    8    6
         2     9       8    9    5    4
         3     9       8   10    3    3
  9      1     2       7    6    8    9
         2     7       7    5    6    8
         3     7       6    5    7    8
 10      1     2       4    8    6    2
         2     3       4    7    5    2
         3     8       3    6    5    2
 11      1     1       7    8    8    2
         2     2       4    7    8    1
         3     5       4    7    4    1
 12      1     6       9    6    4    7
         2     9       8    6    4    6
         3    10       6    6    4    4
 13      1     4       8    4    9    8
         2     9       7    3    9    6
         3     9       7    3    8    7
 14      1     2       9    7    6    7
         2     6       6    4    4    3
         3     6       4    3    6    7
 15      1     1       9    5    7    9
         2     2       8    5    5    5
         3     7       7    5    4    5
 16      1     3       4    7    6    6
         2     6       3    7    4    4
         3     6       4    7    4    3
 17      1     2       6    9    8    7
         2     6       5    7    7    1
         3     6       2    9    6    4
 18      1     7       8    7    2    5
         2     8       7    6    1    3
         3     9       7    4    1    1
 19      1     3       3    6    8    2
         2     5       2    3    6    2
         3     9       1    1    6    2
 20      1     8       4    7    9    8
         2     9       2    5    9    6
         3    10       1    3    9    2
 21      1     5       8    6    7    8
         2     6       6    5    5    6
         3    10       3    4    5    4
 22      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   20   18  114   82
************************************************************************
DEADLINES:
jobnr.  deadline
  2       89
  3       55
  4       135
  5       139
  6       109
  9       141
  12      91
  13      62
  14      156
  15      11
  17      143
  18      78
************************************************************************
