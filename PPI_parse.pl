#!/usr/bin/env perl -wan
BEGIN{ $t=<>;$h={}};
#print join "\t:",$F[0],"\n";
$r= {} unless $h->{$F[0]};
$r->{$F[1]} = $F[15];
$h->{$F[0]}=$r;
END{
@ky = keys %{$h};
print join ",", @ky;
print "\n";
for $k (@ky) {
    $r = $h->{$k};
    @v = map { $r->{$_} or -1}  @ky;
 
    print "@v\n";
}
}
