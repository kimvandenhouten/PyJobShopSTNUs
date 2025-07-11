************************************************************************
file with basedata            : md363_.bas
initial value random generator: 1387329462
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  22
horizon                       :  160
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     20      0       17        2       17
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           6   8  15
   3        3          3           5   7   9
   4        3          3          10  18  21
   5        3          2          10  11
   6        3          1          11
   7        3          3          15  17  21
   8        3          3           9  12  17
   9        3          1          16
  10        3          3          13  14  16
  11        3          1          14
  12        3          3          13  20  21
  13        3          1          19
  14        3          1          17
  15        3          2          16  18
  16        3          1          19
  17        3          2          19  20
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
  2      1     3       6    0    7    4
         2     5       6    0    5    2
         3     5       0    7    7    2
  3      1     1       0    7   10    8
         2     5       0    5    9    7
         3     6       0    3    7    7
  4      1     3       0    7   10    4
         2     6       4    0    6    4
         3    10       0    6    4    4
  5      1     4       4    0    5    6
         2     5       3    0    4    6
         3     6       3    0    1    6
  6      1     6       0    7   10    7
         2     7       0    6    9    7
         3    10       3    0    9    6
  7      1     2       9    0   10    6
         2     2      10    0    7    4
         3    10       9    0    4    3
  8      1     3       2    0    8    1
         2     4       2    0    7    1
         3     6       1    0    6    1
  9      1     1       0    9    8    7
         2     4       2    0    8    6
         3     6       0    9    5    6
 10      1     3       0    8    8    8
         2     6       0    5    6    6
         3    10       5    0    4    5
 11      1     3       5    0    8    6
         2     8       0    6    7    5
         3     9       0    5    6    4
 12      1     5      10    0    9    3
         2     7       8    0    4    3
         3     8       7    0    3    2
 13      1     2       5    0    8    8
         2     6       0    7    8    7
         3     9       0    6    8    7
 14      1     1       4    0    9    6
         2     3       0   10    7    4
         3     8       0    8    4    4
 15      1     2       9    0    5    4
         2     9       0    8    4    3
         3    10       0    8    1    3
 16      1     5       4    0    8    4
         2     7       0    2    5    2
         3     7       3    0    5    2
 17      1     3       0    6    5    7
         2     9       4    0    3    7
         3    10       4    0    2    7
 18      1     2       7    0    3    9
         2     6       0    8    2    5
         3    10       6    0    2    4
 19      1     1       9    0    5    4
         2     2       5    0    4    4
         3     5       2    0    2    3
 20      1     1       7    0    5    9
         2     5       0    4    4    9
         3     8       5    0    1    8
 21      1     2       6    0    6    6
         2     3       3    0    5    5
         3     7       3    0    4    5
 22      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   23   19  115  103
************************************************************************
DEADLINES:
jobnr.  deadline
  2       5
  6       33
  7       158
  12      128
  13      84
  15      96
  16      156
************************************************************************
