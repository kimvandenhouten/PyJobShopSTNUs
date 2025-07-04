************************************************************************
file with basedata            : md382_.bas
initial value random generator: 1888868591
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
    1     20      0       16        5       16
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          1           6
   3        3          2           5  11
   4        3          2           7   8
   5        3          3           6  16  17
   6        3          3          12  13  19
   7        3          1          10
   8        3          3           9  10  11
   9        3          3          12  13  15
  10        3          3          14  15  19
  11        3          3          16  18  21
  12        3          1          14
  13        3          1          21
  14        3          1          18
  15        3          3          16  18  21
  16        3          1          20
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
  2      1     3       5    5    6    8
         2     4       4    3    5    5
         3    10       4    2    5    4
  3      1     5       8    4    6    8
         2     7       7    3    3    8
         3     9       6    2    3    8
  4      1     2       4    7    6    7
         2     5       3    6    5    6
         3     9       2    4    5    4
  5      1     1      10    7   10    1
         2     4       9    4    9    1
         3     6       9    3    9    1
  6      1     3       6    3   10    1
         2     7       3    3    8    1
         3     9       2    3    8    1
  7      1     1       5    6    4    4
         2     2       3    6    4    4
         3    10       3    6    2    3
  8      1     2       3    4    8    9
         2     3       3    4    7    7
         3     4       3    4    5    4
  9      1     3      10    4    9   10
         2     4       7    4    6    5
         3     9       3    3    5    5
 10      1     3       3    5    6    3
         2     8       2    4    3    2
         3     8       1    3    4    3
 11      1     4       8    6    7    7
         2     4       6    7    7    7
         3     6       6    4    5    5
 12      1     4       8    6    6    5
         2     7       7    6    4    3
         3     9       5    5    3    3
 13      1     2       8   10    9    8
         2     6       5    8    6    7
         3     9       3    8    5    5
 14      1     1       7    4    9    8
         2     2       6    4    7    5
         3     8       5    4    5    5
 15      1     1       8    2    7    6
         2     4       6    2    6    4
         3     9       5    2    4    3
 16      1     1       6    5    4    4
         2    10       5    4    2    3
         3    10       5    5    2    1
 17      1     4       9    3    4    6
         2     5       9    2    3    6
         3     8       7    2    1    5
 18      1     1       8    6    5   10
         2     8       6    6    5   10
         3     9       2    6    3   10
 19      1     6      10    5    8    6
         2     8      10    5    7    4
         3    10      10    4    7    1
 20      1     1       4    2    9    8
         2     4       4    2    7    8
         3     6       3    1    7    7
 21      1     4       7    7    8    8
         2     8       6    6    4    4
         3     8       5    7    7    4
 22      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   23   16  141  127
************************************************************************
DEADLINES:
jobnr.  deadline
  2       159
  3       3
  4       65
  5       44
  6       117
  7       144
  8       6
  11      92
  12      85
  14      20
  15      8
  17      133
  19      76
  20      160
************************************************************************
