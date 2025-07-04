************************************************************************
file with basedata            : md351_.bas
initial value random generator: 1054384122
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  22
horizon                       :  161
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     20      0       30        4       30
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           5   7   9
   3        3          3           6  13  16
   4        3          3           6  12  21
   5        3          3          12  17  19
   6        3          1          19
   7        3          3           8  12  13
   8        3          2          10  11
   9        3          2          10  14
  10        3          2          15  17
  11        3          3          14  15  16
  12        3          1          20
  13        3          2          15  21
  14        3          2          17  18
  15        3          1          20
  16        3          1          19
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
  2      1     5       2    7    0    4
         2     5       4    6    0    4
         3     5       7    6    0    3
  3      1     5       5    4   10    0
         2     5       6    4    9    0
         3     9       1    4    8    0
  4      1     1       8    8    0    7
         2     3       8    3    0    7
         3     4       5    1    0    6
  5      1     4       9    8    2    0
         2     7       7    5    0    8
         3    10       6    5    0    2
  6      1     6       3    5    3    0
         2     6       3    5    0    2
         3    10       3    3    3    0
  7      1     3       2    1    0    5
         2     7       1    1    0    5
         3    10       1    1    7    0
  8      1     3       8    2    0    9
         2     7       7    2    6    0
         3     9       7    2    0    9
  9      1     2       7    5    0    3
         2     4       6    5    4    0
         3    10       1    3    0    1
 10      1     5       7    8    0    6
         2     6       6    7    3    0
         3     9       6    7    0    5
 11      1     5       7    9    0    7
         2     7       6    8    0    2
         3     9       3    7    3    0
 12      1     8       4    5    3    0
         2     9       4    4    0    9
         3    10       4    4    0    4
 13      1     5       4    8    4    0
         2     5       4    7    0    5
         3     6       2    3    5    0
 14      1     4       7    4    6    0
         2     6       4    3    0    5
         3     8       2    3    0    1
 15      1     4       9    3    8    0
         2     5       8    3    7    0
         3     5       7    3    8    0
 16      1     1       9    3    0    5
         2     3       9    3    0    4
         3     6       8    2    0    4
 17      1     6       9    9    5    0
         2     8       9    6    4    0
         3    10       7    3    0    9
 18      1     3       7    7    0    3
         2     5       6    6    0    2
         3     8       5    6    0    1
 19      1     2       6    9    0   10
         2     4       5    7    0    7
         3     5       4    6    6    0
 20      1     4       1    6    0    5
         2     5       1    4    9    0
         3     8       1    3    0    4
 21      1     3       6    5   10    0
         2     7       4    4    7    0
         3    10       2    3    0    6
 22      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   24   24   90  108
************************************************************************
DEADLINES:
jobnr.  deadline
  2       125
  3       67
  5       32
  6       119
  7       104
  9       40
  10      113
  11      69
  12      126
  13      54
  14      8
  15      133
  18      146
  19      109
  21      153
************************************************************************
