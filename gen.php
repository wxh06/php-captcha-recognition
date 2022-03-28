<?php
include "vendor/autoload.php";
use Gregwar\Captcha\CaptchaBuilder;

function create_uuid($prefix="") {
    $chars = md5(uniqid(mt_rand(), true));
    $uuid = substr ( $chars, 0, 8 ) . '-'
        . substr ( $chars, 8, 4 ) . '-'
        . substr ( $chars, 12, 4 ) . '-'
        . substr ( $chars, 16, 4 ) . '-'
        . substr ( $chars, 20, 12 );
    return $prefix.$uuid ;
}

for ($i = 1; $i <= 10; ++$i) {
    $builder = new CaptchaBuilder;
    $dir = 'data/';
    $builder->build(90, 35);
    $builder->save($dir . $builder->getPhrase() . '_' . create_uuid() . '.png');
}
