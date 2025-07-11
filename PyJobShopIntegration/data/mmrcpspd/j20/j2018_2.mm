************************************************************************
file with basedata            : md338_.bas
initial value random generator: 2060071247
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  22
horizon                       :  181
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     20      0       27        6       27
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           5   6   7
   3        3          3          10  11  16
   4        3          1           5
   5        3          2           8   9
   6        3          2          11  16
   7        3          3           9  12  14
   8        3          3          10  17  19
   9        3          2          10  11
  10        3          1          20
  11        3          1          13
  12        3          2          13  16
  13        3          2          15  17
  14        3          2          19  21
  15        3          3          18  19  21
  16        3          1          17
  17        3          2          20  21
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
  2      1     8       0    4    0    4
         2     9       0    3    0    3
         3     9       0    1    4    0
  3      1     4       0    6    0    6
         2     8       0    5    3    0
         3     9       9    0    0    5
  4      1     1       0    9    0    8
         2     6       0    7    5    0
         3     9       0    3    0    3
  5      1     1       0    1    5    0
         2     6       7    0    5    0
         3    10       7    0    3    0
  6      1     4       0    3    2    0
         2    10       5    0    0    2
         3    10       8    0    1    0
  7      1     1       0    4    7    0
         2     3      10    0    6    0
         3     9       5    0    6    0
  8      1     1       3    0    0    5
         2    10       3    0    0    3
         3    10       0   10    0    3
  9      1     1       0    7    8    0
         2     5       0    6    0    6
         3     6       0    5    1    0
 10      1     2       0   10    0    7
         2     4       2    0    0    7
         3    10       0    9    0    6
 11      1     5       5    0    8    0
         2     6       4    0    7    0
         3     8       0    4    7    0
 12      1     4       0    9    8    0
         2     6       0    7    0    8
         3     7       0    6    0    4
 13      1     1       0    8    0    4
         2     4       0    6    0    1
         3     9       0    5    3    0
 14      1     7       2    0    8    0
         2    10       0    9    2    0
         3    10       2    0    2    0
 15      1     1       2    0    7    0
         2     8       2    0    0   10
         3    10       0    8    0    7
 16      1     1       0    5    5    0
         2     5       0    4    0    7
         3    10       0    3    3    0
 17      1     3       0    7    0    8
         2     4       9    0    3    0
         3     9       0    4    0    7
 18      1     3       0    9    2    0
         2     9       0    5    2    0
         3    10       0    3    1    0
 19      1     8       0    5    0    3
         2     8       7    0    3    0
         3     9       5    0    1    0
 20      1     1       0    8    0    6
         2     1       0    7    0    7
         3     7       0    6    0    6
 21      1     5       0    1    0    6
         2     9       9    0    8    0
         3    10       2    0    0    5
 22      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   17   22   72   72
************************************************************************
DEADLINES:
jobnr.  deadline
  7       91
  9       110
  12      61
  15      77
  17      87
  19      155
************************************************************************
