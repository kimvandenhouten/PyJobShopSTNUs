************************************************************************
file with basedata            : md346_.bas
initial value random generator: 1225824544
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  22
horizon                       :  153
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     20      0       27       10       27
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3          11  12  18
   3        3          1          12
   4        3          3           5   6   7
   5        3          3           8   9  16
   6        3          3          11  12  13
   7        3          3           8  15  19
   8        3          1          20
   9        3          3          10  11  13
  10        3          1          17
  11        3          2          15  20
  12        3          1          14
  13        3          2          14  19
  14        3          2          15  17
  15        3          1          21
  16        3          2          18  21
  17        3          2          20  21
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
  2      1    10       7    0    0    3
         2    10       3    0    0    5
         3    10       0    5    5    0
  3      1     5       0    5    0    2
         2     5       0    5    2    0
         3     7       8    0    1    0
  4      1     7       0    7    4    0
         2     7       2    0    0    7
         3    10       0    8    0    6
  5      1     3       7    0    8    0
         2     4       0    6    0    6
         3    10       0    6    8    0
  6      1     6       0    7    8    0
         2     8       1    0    0    8
         3     9       0    4    0    5
  7      1     7      10    0    5    0
         2     8       0    9    3    0
         3    10       8    0    0    6
  8      1     3       0    9    0   10
         2     5       0    6    9    0
         3     9       0    4    0    7
  9      1     1       0    6    0    8
         2     6       0    6    8    0
         3    10       0    6    6    0
 10      1     3       0    5    0    8
         2     5       8    0    0    4
         3     6       0    1    6    0
 11      1     4       0    3    5    0
         2     9       0    2    0    8
         3     9       0    3    0    7
 12      1     4       5    0    7    0
         2     5       5    0    6    0
         3     7       4    0    6    0
 13      1     2       6    0    0    7
         2     8       0    9    0    6
         3    10       0    6    0    6
 14      1     1       0    5    0    3
         2     2       2    0    0    3
         3     2       0    5    2    0
 15      1     2       0   10    4    0
         2     4       0    9    0    9
         3     4       4    0    3    0
 16      1     1       6    0    0   10
         2     7       5    0    0    9
         3     7       0    9    4    0
 17      1     5       6    0    6    0
         2     5       0    9    4    0
         3     6       0    8    0    4
 18      1     2       7    0    7    0
         2     5       6    0    4    0
         3     5       0    9    4    0
 19      1     5       0    8    9    0
         2     6       5    0    7    0
         3     9       0    7    5    0
 20      1     4       0   10    0    4
         2     4       9    0    0    4
         3     6       0   10    1    0
 21      1     2       0    9    0    5
         2     4       0    4    3    0
         3     7       7    0    2    0
 22      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   15   21  103  110
************************************************************************
DEADLINES:
jobnr.  deadline
  2       136
  3       9
  4       106
  5       52
  6       66
  8       26
  9       48
  10      41
  11      16
  13      42
  14      10
  17      103
  18      117
  20      42
  21      28
************************************************************************
