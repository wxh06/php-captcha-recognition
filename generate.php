<?php

require 'vendor/autoload.php';

use Gregwar\Captcha\CaptchaBuilder;

while (1) {
  $builder = new CaptchaBuilder(4);
  $builder->build($width = 90, $height = 35);
  print($builder->getPhrase());
  $builder->output();
}
