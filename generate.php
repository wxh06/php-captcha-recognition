<?php

error_reporting(0);

require 'vendor/autoload.php';

use Gregwar\Captcha\CaptchaBuilder;

while (1) {
  $builder = new CaptchaBuilder(intval($argv[1]));
  $builder->build($width = intval($argv[2]), $height = intval($argv[3]));
  print($builder->getPhrase());
  $builder->output();
}
