************************************************************************
file with basedata            : md336_.bas
initial value random generator: 207519839
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  22
horizon                       :  173
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     20      0       36       16       36
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           5   9  15
   3        3          2           7  11
   4        3          3           5   7  15
   5        3          3           6  10  11
   6        3          3           8  14  16
   7        3          2           8  14
   8        3          2          17  21
   9        3          3          10  11  16
  10        3          3          19  20  21
  11        3          1          12
  12        3          2          13  19
  13        3          1          14
  14        3          1          17
  15        3          2          18  19
  16        3          1          21
  17        3          1          18
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
  2      1     3       8    6    7    0
         2     8       8    6    0    7
         3     8       6    6    3    0
  3      1     9       4    5    0    6
         2     9       4    7    2    0
         3    10       3    5    0    7
  4      1     2       2    7    0    8
         2     4       1    7    0    7
         3     9       1    6    0    7
  5      1     3       8   10    7    0
         2     4       7   10    0    3
         3     7       5    9    0    1
  6      1     1      10   10    0    8
         2     6       9    8    0    6
         3     9       8    8    0    4
  7      1     3       4    9    7    0
         2     6       2    8    5    0
         3     8       2    6    2    0
  8      1     3       7    7    7    0
         2     5       6    6    0    6
         3    10       4    3    0    5
  9      1     3       8    6    0    6
         2     8       6    4    0    6
         3     8       7    5    7    0
 10      1     2       9    8    5    0
         2     3       4    6    0    4
         3     8       4    1    4    0
 11      1     6       6    9    6    0
         2     8       4    9    0    8
         3     8       2    8    3    0
 12      1     4       9   10    0    7
         2     7       5    5    5    0
         3     9       2    4    0    7
 13      1     2       6    5    8    0
         2     3       6    4    8    0
         3    10       2    2    7    0
 14      1     1       8    7    0    6
         2     5       5    3    9    0
         3    10       2    2    0    5
 15      1     1       7    7    0   10
         2     6       3    5    0    9
         3     9       1    2    0    8
 16      1     2      10    3    0    4
         2     3       9    2    0    4
         3     6       9    2    0    3
 17      1     3       3    7    4    0
         2     6       3    7    3    0
         3     8       2    7    0    6
 18      1     5       8    9    0    8
         2     6       8    6    0    8
         3     8       4    3    0    8
 19      1     3       6    9    0    6
         2     6       5    8    0    5
         3     9       1    7    6    0
 20      1     6       8    4    2    0
         2     6       8    4    0    8
         3     9       5    4    0    4
 21      1     3       9    8    7    0
         2     8       8    7    5    0
         3    10       6    5    5    0
 22      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   27   30   52   71
************************************************************************
DEADLINES:
jobnr.  deadline
  2       147
  3       54
  4       149
  6       118
  7       161
  8       64
  9       106
  10      19
  11      52
  13      106
  14      157
  15      139
  16      31
  18      27
  19      131
  20      163
  21      137
************************************************************************
