# Vcpkg support

This directory is meant to provide the sources for the official [vcpkg port] 
of palloc, but can also be used to override the official port with
your own variant.

For example, you can edit the [`portfile.cmake`](portfile.cmake) 
to check out a specific commit, version, or branch of palloc, or set further options. 
You can install such custom port as:

```sh
$ vcpkg install "palloc[override]" --recurse --overlay-ports=./contrib/vcpkg
```

This will also show the correct sha512 hash if you use a custom version.
Another way is to refer to the overlay from the [vcpkg-configuration.json](https://learn.microsoft.com/en-us/vcpkg/reference/vcpkg-configuration-json) file.
See also the vcpkg [documentation](https://learn.microsoft.com/en-us/vcpkg/produce/update-package-version) for more information.


# Using palloc from vcpkg

When using [cmake with vcpkg](https://learn.microsoft.com/en-us/vcpkg/get_started/get-started?pivots=shell-powershell), 
you can use palloc from the `CMakeLists.txt` as:

```cmake
find_package(palloc CONFIG REQUIRED)
target_link_libraries(main PRIVATE palloc)
```

See [`test/CMakeLists.txt](../../test/CMakeLists.txt) for more examples.


# Acknowledgements

The original port for vckpg was contributed by many people, including: @vicroms, @myd7349, @PhoubeHui, @LilyWangL,
@JonLiu1993, @RT2Code, Remy Tassoux, @wangao, @BillyONeal, @jiayuehua, @dg0yt, @gerar-ryan-immersaview, @nickdademo, 
and @jimwang118 -- Thank you so much!


[vcpkg port]: https://github.com/microsoft/vcpkg/tree/master/ports/palloc
