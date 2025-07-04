************************************************************************
file with basedata            : md375_.bas
initial value random generator: 1867233409
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
    1     20      0       26       12       26
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          1          14
   3        3          3           5   9  16
   4        3          3           5   6  18
   5        3          2           8  15
   6        3          3           7  12  15
   7        3          2           8  16
   8        3          2          10  13
   9        3          3          11  13  18
  10        3          2          11  14
  11        3          3          19  20  21
  12        3          2          13  19
  13        3          1          17
  14        3          1          20
  15        3          2          17  21
  16        3          2          17  21
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
  2      1     5       8    5    7    5
         2     6       8    3    4    4
         3     6       8    4    4    3
  3      1     2       8   10    8    5
         2     8       7    9    6    4
         3    10       5    7    2    2
  4      1     4       6    6    3    9
         2     5       4    5    2    8
         3     8       4    3    2    8
  5      1     1      10    7    9    9
         2     3       8    5    8    4
         3    10       6    5    8    1
  6      1     6      10   10    2    9
         2     7      10    8    1    9
         3    10       9    7    1    9
  7      1     1       2    1    8    5
         2     2       2    1    7    5
         3     3       1    1    7    4
  8      1     1       4    8    5    5
         2     8       4    6    5    4
         3    10       3    5    4    4
  9      1     1       3    4    7    4
         2     3       2    3    7    4
         3     8       1    2    7    4
 10      1     5       8    5    6    5
         2     6       7    4    6    3
         3     6       8    4    4    2
 11      1     1       9    9    4    3
         2     4       8    6    3    2
         3    10       7    3    3    1
 12      1     7       5    2    8    5
         2     8       4    2    5    4
         3    10       4    1    2    3
 13      1     2       8    6    9    9
         2     4       7    5    8    7
         3     6       5    5    6    6
 14      1     1       9    4    6    9
         2     5       8    4    4    8
         3     7       6    4    4    5
 15      1     6       9    4    6    5
         2     7       9    4    5    4
         3     8       9    3    4    4
 16      1     3       5    7    8    3
         2     4       3    4    7    3
         3     5       3    3    6    3
 17      1     2       9   10    7    9
         2     4       6    6    5    5
         3     7       5    5    2    3
 18      1     4       3    6    9    6
         2     6       1    6    7    2
         3     6       1    3    8    5
 19      1     1       9    6    6   10
         2     7       8    6    5   10
         3    10       7    4    2   10
 20      1     5       4    9    8    8
         2     6       3    8    8    5
         3     7       1    7    8    2
 21      1     4       1    8    3    7
         2     7       1    4    3    5
         3    10       1    2    3    4
 22      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   26   23  118  118
************************************************************************
DEADLINES:
jobnr.  deadline
  2       26
  3       39
  4       29
  5       80
  6       149
  7       20
  8       35
  9       74
  10      155
  11      68
  12      98
  13      115
  14      61
  15      81
  16      71
  17      119
  18      83
  19      68
  20      83
************************************************************************
