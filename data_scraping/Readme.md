1, configure login information

	1) put the username and password of the Instagram account in credential.txt at the first and second line respectively.


2, search according to hashtags

	1) put all hashtags to search in each line of the hashtags.txt file.
	
	2) run command: php food_hashtag_search.php

3, search according to locations

	1) put all geo-bounding boxes in each line of multi_bb.txt, bounding boxes information includes (latitude of the upper-left corner, longitude of the upper-left corner, latitude of the bottom-right corner, longitude of the bottom-right corner)
	
	2) run command: php search_all_location.php
	
	3) Instagram Position ID will be saved in location_xx_xx_xx_xx.json file for every bounding box.
	
	for every file saving Instagram Position ID:
	
		4) put the name of the file containing Instagram Position ID at the line 19 of the search_by_location.php file to specify the location to scrape and run: php search_by_location.php

