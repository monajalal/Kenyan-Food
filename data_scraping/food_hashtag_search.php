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

$file = fopen("hashtags.txt",'r');
$hashtags = array();
if(!$file){
    return 'file open fail';
}else{
    $i = 0;
    while (1){
        $str = fgets($file); // read each lines of file in hashtags.txt
        $str = strtolower(str_replace(array(" ","\r\n", "\r", "\n"), "", $str)); // squish out \n or other possible interruption
        if($str) {
            $hashtags[$i] = $str; // store if valid
            $i++ ;
        }
        else{
            break; // break when meet end
        }
    }
    fclose($file);
}


$ig = new \InstagramAPI\Instagram($debug, $truncatedDebug); // initiate module

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
    $ig->login($username, $password); // login
} catch (\Exception $e) {
    echo 'Something went wrong: '.$e->getMessage()."\n";
    exit(0);
}


foreach ((array)$hashtags as $tag) {

    try {
        $rankToken = \InstagramAPI\Signatures::generateUUID(); //get UUID 
        $maxId = null;
        $thres = 10000;
        $out = array();
        $num = 0;
        do {
            $response = $ig->hashtag->getFeed($tag, $rankToken, $maxId); // get response
            $j = json_decode($response); // resolve json string
            foreach ($j->ranked_items as $item) {
                if (1) {
                    $num = $num + 1; // record result number
                    if ($item->carousel_media_count != NULL) // if have multiple images in a post
                    {
                        $image_urls = array();
                        $counter = 1;
                        foreach ($item->carousel_media as $i) 
                        {
                            array_push($image_urls, ($i->image_versions2->candidates)[0]->url); // get images' url in to a list
                            getImage(($i->image_versions2->candidates)[0]->url,$tag."/",$item->pk."_".(string)$counter.".jpg");
                            $counter = $counter + 1;
                        }
                        $output = array("pk"=>($item->pk),"id"=>($item->id),"image_url"=>$image_urls,"text"=>($item->caption->text),"post_url"=>("https://instagram.com/p/".$item->code)."/"); // record pk, id , etc..
                        var_dump($output);
                        array_push($out, $output);
                    }
                    else
                    {
                        $output = array("pk"=>($item->pk),"id"=>($item->id),"image_url"=>(($item->image_versions2->candidates)[0]->url),"text"=>($item->caption->text),"post_url"=>("https://instagram.com/p/".$item->code)."/"); // record pk, id , etc..
                        var_dump($output);
                        array_push($out, $output);
                        getImage(($item->image_versions2->candidates)[0]->url,$tag."/",$item->pk.".jpg");

                    }
                }
            }
            foreach ($j->items as $item) {
                if (1) {
                    $num = $num + 1; // record result number
                    if ($item->carousel_media_count != NULL) 
                    {
                        $image_urls = array();
                        $counter = 1;
                        foreach ($item->carousel_media as $i) 
                        {
                            array_push($image_urls, ($i->image_versions2->candidates)[0]->url);
                            getImage(($i->image_versions2->candidates)[0]->url,$tag."/",$item->pk."_".(string)$counter.".jpg");
                            $counter = $counter + 1;
                        }
                        $output = array("pk"=>($item->pk),"id"=>($item->id),"image_url"=>$image_urls,"text"=>($item->caption->text),"post_url"=>("https://instagram.com/p/".$item->code)."/"); // record pk, id , etc..
                        var_dump($output);
                        array_push($out, $output);
                    }
                    else
                    {
                        $output = array("pk"=>($item->pk),"id"=>($item->id),"image_url"=>(($item->image_versions2->candidates)[0]->url),"text"=>($item->caption->text),"post_url"=>("https://instagram.com/p/".$item->code)."/"); // record pk, id , etc..
                        var_dump($output);
                        array_push($out, $output);
                        getImage(($item->image_versions2->candidates)[0]->url,$tag."/",$item->pk.".jpg");

                    }
                }
            }
            // foreach ((array)$response->getSections() as $section) 
            // {
            //     $medias = ($section->getLayoutContent()->getMedias());
            //     foreach((array)$medias as $media)
            //     {
            //         $id = $media->getMedia()->getId();
            //         $pk = $media->getMedia()->getPk();
            //         $text = "";
            //         if($media->getMedia()->getCaption() != null) {
            //             $text = $media->getMedia()->getCaption()->getText();
            //         }
            //         if($media->getMedia()->getCarouselMedia() != null) {
            //             $im_url = array();
            //             $counter = 1;
            //             foreach((array)$media->getMedia()->getCarouselMedia() as $c) {
            //                 $num += 1;
            //                 $array_push($im_url,$c->getImageVersions2()->getCandidates()[0]->getUrl());
            //                 getImage($im_url,$tag."/",$pk."_".strval($counter).".jpg");
            //                 $counter += 1;
            //             }
            //         }
            //         else {
            //             $num += 1;
            //             $im_url = $media->getMedia()->getImageVersions2()->getCandidates()[0]->getUrl();
            //             getImage($im_url,$tag."/",$pk.".jpg");
            //         }
            //         $post_url = $media->getMedia()->getItemUrl();
            //         $output = array("pk"=>$pk,"id"=>$id,"image_url"=>$im_url,"text"=>$text,"post_url"=>$post_url);
            //         var_dump($output);
            //         array_push($out, $output);
            //     }
            // }

            foreach ($response->getItems() as $item) {
                printf("[%s] https://instagram.com/p/%s/\n", $item->getId(), $item->getCode()); // print post's url
            }
            $maxId = $response->getNextMaxId(); // get next maxId to paginate
            if ($thres != 0) // check if exceed needed number of post, do not check if $thres == 0
            {
                if ($num >= $thres)
                {
                    break;
                }
            }
            echo "Sleeping for 5s...\n";
            sleep(5);
        } while ($maxId !== null); // Must use "!==" for comparison instead of "!=".
        $file=fopen("output_food_".$tag.".json","w"); 
        $j = json_encode($out);// convert to json string 
        fwrite($file, $j); // store it
        fclose($file);
    }
catch (\Exception $e) {
    echo 'Something went wrong: '.$e->getMessage()."\n";
}
}
