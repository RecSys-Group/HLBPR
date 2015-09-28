This is a static methods. The items recommended to different users are the same. We select the most popular N item, and recommend to the users.

input:

1.item.txt: This file is the set of items
format:
---------
,0
iid(int),item(int)
---------
2.users.txt:This file is the set of users
format:
---------
,0
uid(int),user(int)
---------
3.new_ui.txt:This file is the users' transaction records, number_of_times indicate how many times the user purchased the item
format:
---------
,0,1,2
line,uid(float),iid(float),number_of_times(float)
---------
4.newtest.txt:This file is used to test topN 
format:
---------
,0,1,...
uid(int),item1,item2,...
---------

