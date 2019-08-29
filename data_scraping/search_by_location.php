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

$file_name = "0.001_1.7_38.3_40.979.json";
$file = fopen("location_".$file_name,'r');
if(!$file){
    return 'file open fail';
}else{
    $str = fgets($file); 
    $loc = json_decode($str);
var_dump($loc);
    fclose($file);
}
var_dump($loc);
try {
    $ig->login($username, $password); // login
} catch (\Exception $e) {
    echo 'Something went wrong: '.$e->getMessage()."\n";
    exit(0);
}

function getImage($url,$save_dir='',$filename='',$type=0){ // download image according to url
    if(trim($url)==''){
        return array('file_name'=>'','save_path'=>'','error'=>1);
    }
    if(trim($save_dir)==''){
        $save_dir='./';
    }
    if(trim($filename)==''){
        $ext=strrchr($url,'.');
        if($ext!='.gif'&&$ext!='.jpg'){
            return array('file_name'=>'','save_path'=>'','error'=>3);
        }
        $filename=time().$ext;
    }
    if(0!==strrpos($save_dir,'/')){
        $save_dir.='/';
    }
    if(!file_exists($save_dir)&&!mkdir($save_dir,0777,true)){
        return array('file_name'=>'','save_path'=>'','error'=>5);
    }
    if($type){
        $ch=curl_init();
        $timeout=300;
        curl_setopt($ch,CURLOPT_URL,$url);
        curl_setopt($ch,CURLOPT_RETURNTRANSFER,1);
        curl_setopt($ch,CURLOPT_CONNECTTIMEOUT,$timeout);
        $img=curl_exec($ch);
        curl_close($ch);
    }else{
        ob_start(); 
        readfile($url);
        $img=ob_get_contents(); 
        ob_end_clean(); 
    }
    $fp2=@fopen($save_dir.$filename,'a');
    fwrite($fp2,$img);
    fclose($fp2);
    unset($img,$url);
    return array('file_name'=>$filename,'save_path'=>$save_dir.$filename,'error'=>0);
}

try {

    $rankToken = null; // set up UUID

    foreach ((array)$loc as $l) {
        $rankToken = \InstagramAPI\Signatures::generateUUID();
        $file=fopen("temp_loc_output_".$file_name,"a");

        $maxId = null;
        //$out = array();
    	$nextMediaIds = null;
    	$nextPage = null;
        do {
            var_dump($l);
	    try{
                $response = $ig->location->getFeed($l, $rankToken,'recent', $nextMediaIds, $nextPage , $maxId); // get post for each location 
    	    }
	    catch(\Exception $e) {
                echo "Exception, Sleeping for 5s...\n";
                echo 'Something went wrong: '.$e->getMessage()."\n";
                sleep(5);
		continue;
            }

	    foreach ((array)$response->getSections() as $section) 
    	    {
        		$medias = ($section->getLayoutContent()->getMedias());
        		foreach((array)$medias as $media)
        		{
        		    $id = $media->getMedia()->getId();
                    $pk = $media->getMedia()->getPk();
                    $lat = $media->getMedia()->getLat();
		    $lng = $media->getMedia()->getLng();
	            $text = "";
                    if($media->getMedia()->getCaption() != null) {
                        $text = $media->getMedia()->getCaption()->getText();
        		    }
        		    if($media->getMedia()->getCarouselMedia() != null) {
                        $im_url = array();
            			foreach((array)$media->getMedia()->getCarouselMedia() as $c) {
            			    array_push($im_url,$c->getImageVersions2()->getCandidates()[0]->getUrl());
                        }
        		    }
        		    else {
            			$im_url = $media->getMedia()->getImageVersions2()->getCandidates()[0]->getUrl();
        		    }
        		    $post_url = $media->getMedia()->getItemUrl();

                    $output = array("pk"=>$pk,"id"=>$id,"image_url"=>$im_url,"text"=>$text,"post_url"=>$post_url,"lat"=>$lat,"lng"=>$lng);
                    var_dump($output);
                    //array_push($out, $output);
    		    $j = json_encode($output);
		    fwrite($file, $j);
		    fwrite($file, "\n");
		    }
    	    }
            $maxId = $response->getNextMaxId(); // get maxId to paginate
            $nextPage = $response->getNextPage();
            $nextMediaIds = $response->getNextMediaIds();
            var_dump($maxId);
            var_dump($nextPage);
            var_dump($nextMediaIds);

	    if(count($nextMediaIds)==0) {
	        $nextMediaIds = null;
	    }
	    
	    echo "Sleeping for 5s...\n";
            sleep(5);

        }while ($maxId !== null); // Must use "!==" for comparison instead of "!=".
        /*if(count($out) > 0) {
            $j = json_encode($out); // encode into json string
            fwrite($file, $j);// store it
            fwrite($file, "\n");
	}*/
        fclose($file);
    }
} catch (\Exception $e) {
    echo 'Something went wrong: '.$e->getMessage()."\n";
}
