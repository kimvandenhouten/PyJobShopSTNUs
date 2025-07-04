************************************************************************
file with basedata            : md362_.bas
initial value random generator: 191123787
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  22
horizon                       :  163
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     20      0       33       13       33
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          2           5   6
   3        3          3           5   8  16
   4        3          2          11  14
   5        3          2           9  20
   6        3          3           7   8  14
   7        3          3          11  13  16
   8        3          1          17
   9        3          3          10  11  19
  10        3          2          12  18
  11        3          2          12  18
  12        3          1          21
  13        3          2          15  18
  14        3          1          20
  15        3          2          17  19
  16        3          3          17  19  21
  17        3          1          20
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
  2      1     6       5    0    3    2
         2     6       0    7    3    3
         3    10       0    2    2    2
  3      1     1       3    0    5    7
         2     7       0    7    5    6
         3     9       0    1    3    2
  4      1     6       5    0    7    2
         2    10       0    9    7    1
         3    10       0    8    7    2
  5      1     5       7    0    6    9
         2     6       0    4    6    6
         3     6       0    3    5    8
  6      1     5       5    0    4    7
         2     5       0    5    5   10
         3     6       0    2    3    6
  7      1     5       0    6    7    3
         2     5       8    0    7    2
         3     9       0    4    6    1
  8      1     1       0    9    8    4
         2     7       4    0    5    3
         3     8       0    9    4    2
  9      1     4       0    2    7    8
         2     7       0    2    4    7
         3     9       5    0    3    5
 10      1     3       0    2    5    2
         2     4       8    0    3    2
         3     7       6    0    1    2
 11      1     2       0    3    4    3
         2     3       6    0    4    3
         3     4       0    3    2    1
 12      1     6       0    2    7   10
         2     7       6    0    6    9
         3     8       4    0    6    9
 13      1     1       3    0    7    8
         2     3       0    2    5    8
         3     4       3    0    5    7
 14      1     7       3    0    5    9
         2     8       0    4    5    6
         3    10       0    3    5    6
 15      1     3       2    0    7    9
         2     7       0    6    4    7
         3    10       0    5    4    6
 16      1     8       0    7    8    4
         2     9       0    7    8    2
         3     9       2    0    7    2
 17      1     5       0    7    5    5
         2     6       9    0    4    4
         3     7       0    5    4    1
 18      1     1       0    7    4    9
         2     2       7    0    4    7
         3    10       0    3    3    4
 19      1     3       9    0    6    2
         2     7       7    0    6    2
         3    10       4    0    5    2
 20      1     4       0    8    9    6
         2     4       0    8   10    4
         3     8      10    0    8    3
 21      1     2       3    0    6    7
         2     4       3    0    6    6
         3     9       0    5    6    4
 22      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   18   12  106   96
************************************************************************
DEADLINES:
jobnr.  deadline
  3       34
  4       114
  7       67
  18      162
************************************************************************
