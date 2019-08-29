<?php

set_time_limit(0);
date_default_timezone_set('UTC');

require __DIR__.'/../vendor/autoload.php';

/////// CONFIG ///////
$file=fopen("credential.txt","r");
$username = fgets($file);
$password = fgets($file);
fclose($file);
$debug = true;
$truncatedDebug = false;
//////////////////////

$ig = new \InstagramAPI\Instagram($debug, $truncatedDebug); // initiate module

$file = fopen("multi_bb.txt",'r');
$bb = array();
if(!$file){
    return 'file open fail';
}
else{
    while (1){
        $str = fgets($file); 
        if($str) {
            $str = str_replace(array("\r\n", "\r", "\n"), "", $str); // squish out \n or other possible interruption
            list($s1, $s2, $s3, $s4) = explode(",",$str);
            $s1 = floatval($s1);
            $s2 = floatval($s2);
            $s3 = floatval($s3);
            $s4 = floatval($s4);
            $lat_small = min($s1,$s3);
            $lat_large = max($s1,$s3);
            $lng_small = min($s2,$s4);
            $lng_large = max($s2,$s4);
            $bb[] = array($lat_small,$lat_large,$lng_small,$lng_large);  // store if valid
        }
        else{
            break; // break when meet end
        }
    }
    fclose($file);
}
function in_bb($lat,$lng,$bb,$x) { // in_bb(0,35,$bb)
    
    if($bb[$x][0] <= $lat && $bb[$x][1] >= $lat && $bb[$x][2] <= $lng && $bb[$x][3] >= $lng) {
    
	print(strval($lat)."-".strval($lng)."-".strval($bb[$x][0])."-".strval($bb[$x][1])."-".strval($bb[$x][2])."-".strval($bb[$x][3]).",in"."\n");
	return true;
    }
    print(strval($lat)."-".strval($lng)."-".strval($bb[$x][0])."-".strval($bb[$x][1])."-".strval($bb[$x][2])."-".strval($bb[$x][3]).",out"."\n");
    return false;
}

try {
    $ig->login($username, $password); // login
} catch (\Exception $e) {
    echo 'Something went wrong: '.$e->getMessage()."\n";
    exit(0);
}

try {
    $rankToken = null; // set up UUID
    $stride = 0.02;

    for($x=0;$x < count($bb);$x++) {
        $file = fopen("location_".strval($bb[$x][0])."_".strval($bb[$x][1])."_".strval($bb[$x][2])."_".strval($bb[$x][3]).".json", "w");
        $loc = array(); // initiate list of location
        for($y = $bb[$x][0];$y < $bb[$x][1];$y+=$stride) {
            for($z = $bb[$x][2];$z < $bb[$x][3];$z+=$stride) {
                $check = array();
                do {
                    try{
                    $response = $ig->location->findPlacesNearby($y,$z,null,$check,$rankToken); // find location nearby Nairobi
                    }
                    catch(\Exception $e) {
                        echo "Exception, Sleeping for 5s...\n";
                        echo 'Something went wrong: '.$e->getMessage()."\n";
                        sleep(5);

                    }
                    $j = json_decode($response);
                    foreach ($j->items as $item) {
                        if(!in_array($item->location->facebook_places_id,$check)) { // add to check list either way
                            array_push($check,$item->location->facebook_places_id);
                        }
                        if(!in_array($item->location->facebook_places_id,$loc)){
			    if($item->location->lat != null && $item->location->lng != null) {
                                if(in_bb($item->location->lat,$item->location->lng,$bb,$x)) {// record if it is in geo bounding box
                                    array_push($loc,$item->location->facebook_places_id);
                                }
			    }
			}
                    }
                    $rankToken = $j->rank_token; // get rank_token to paginate
                    var_dump($loc);
                    echo "Sleeping for 5s...\n";
                    sleep(5);
                } while ($response->getHasMore()); // Must use "!==" for comparison instead of "!=".

            }
        }
        if(count($loc) > 0) {
            $j = json_encode($loc); // encode into json string
            fwrite($file, $j);// store it
        }
        fclose($file);


    }

} catch (\Exception $e) {
    echo 'Something went wrong: '.$e->getMessage()."\n";
    echo 'Something went wrong: code '.$e->getCode()."\n";
}



