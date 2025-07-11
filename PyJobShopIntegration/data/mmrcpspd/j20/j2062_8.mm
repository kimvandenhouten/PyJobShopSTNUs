************************************************************************
file with basedata            : md382_.bas
initial value random generator: 220826849
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
    1     20      0       18        7       18
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           6   7  19
   3        3          2          16  17
   4        3          1           5
   5        3          3           8  12  13
   6        3          3          10  12  14
   7        3          3           9  10  13
   8        3          3           9  10  11
   9        3          2          15  21
  10        3          1          18
  11        3          3          16  18  19
  12        3          1          20
  13        3          3          15  16  21
  14        3          2          17  21
  15        3          1          17
  16        3          1          20
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
  2      1     4       8    6    2    6
         2     9       6    3    1    6
         3    10       2    2    1    6
  3      1     1       4    8    7    7
         2     3       3    8    7    7
         3     8       2    7    7    5
  4      1     3       2    5    6    6
         2     7       2    2    6    5
         3     7       1    3    6    6
  5      1     1       3    5    8    5
         2     3       2    4    8    4
         3     7       2    4    7    2
  6      1     2       3    3    5   10
         2     4       3    3    5    8
         3     7       3    2    2    4
  7      1     2       5    3    3    9
         2     6       4    3    3    8
         3    10       4    3    1    6
  8      1     1       6    5    8    5
         2     5       5    5    6    4
         3     7       3    3    4    2
  9      1     2       9    7    9    5
         2     6       6    6    8    5
         3     9       3    6    8    3
 10      1     5       4    7    6    7
         2     7       4    5    2    2
         3     7       3    7    3    6
 11      1     3       7    8    8    3
         2     5       6    8    7    2
         3    10       3    6    5    1
 12      1     5       6    7   10    4
         2     6       4    5    7    4
         3     7       2    4    7    3
 13      1     4       8    5    8    3
         2     4       9    5    7    3
         3    10       6    4    5    2
 14      1     3      10    7   10    9
         2     5       7    3    7    8
         3     7       3    3    7    6
 15      1     1       4    9    7   10
         2     7       4    6    6    9
         3     9       4    6    2    9
 16      1     2       5    8    5    3
         2     7       5    5    5    3
         3     9       4    1    3    2
 17      1     3       7    9    7    8
         2     9       6    8    3    7
         3    10       3    8    3    6
 18      1     2       2    7    9    7
         2     3       1    7    8    7
         3     9       1    2    6    5
 19      1     1      10    8    8    7
         2     2      10    4    6    7
         3     4      10    4    5    5
 20      1     2       4    9    7    4
         2     9       3    8    4    3
         3    10       1    6    3    2
 21      1     3       7    9    7    7
         2     3       7    7    6    8
         3     9       6    5    4    5
 22      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   28   25  140  126
************************************************************************
DEADLINES:
jobnr.  deadline
  3       42
  5       34
  6       87
  7       151
  9       130
  10      24
  12      131
  14      112
  16      50
  20      157
************************************************************************
