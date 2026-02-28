# Windows Override

<span id="override_on_windows">We use a separate redirection DLL to override palloc on Windows</span> 
such that we redirect all malloc/free calls that go through the (dynamic) C runtime allocator, 
including those from other DLL's or libraries. As it intercepts all allocation calls on a low level, 
it can be used on large programs that include other 3rd party components.
There are four requirements to make the overriding work well:

1. Use the C-runtime library as a DLL (using the `/MD` or `/MDd` switch).

2. Link your program explicitly with the `palloc.dll.lib` export library for
   the `palloc.dll` -- which contains all palloc functionality.
   To ensure the `palloc.dll` is actually loaded at run-time it is easiest 
   to insert some call to the palloc API in the `main` function, like `palloc_version()`
   (or use the `/include:palloc_version` switch on the linker, or
   similarly, `#pragma comment(linker, "/include:palloc_version")` in some source file). 
   See the `palloc-test-override` project for an example on how to use this. 

3. The `palloc-redirect.dll` must be put in the same folder as the main 
   `palloc.dll` at runtime (as it is a dependency of that DLL).
   The redirection DLL ensures that all calls to the C runtime malloc API get 
   redirected to palloc functions (which reside in `palloc.dll`).

4. Ensure the `palloc.dll` comes as early as possible in the import
   list of the final executable (so it can intercept all potential allocations).
   You can use `minject -l <exe>` to check this if needed.

```csharp
┌──────────────┐                                                    
│ Your Program │                                                    
└────┬─────────┘                                                    
     │                                                              
     │ palloc_version()  ┌───────────────┐     ┌───────────────────────┐
     ├──────────────►│ palloc.dll  ├────►│ palloc-redirect.dll │
     │               └──────┬────────┘     └───────────────────────┘
     │                      ▼                                       
     │ malloc() etc. ┌──────────────┐                               
     ├──────────────►│ ucrtbase.dll │                               
     │               └──────────────┘                               
     │                                                              
     │                                                              
     └──────────────► ...                                           
```

For best performance on Windows with C++, it
is also recommended to also override the `new`/`delete` operations (by including
[`palloc-new-delete.h`](../include/palloc-new-delete.h) 
a single(!) source file in your project).

The environment variable `PALLOC_DISABLE_REDIRECT=1` can be used to disable dynamic
overriding at run-time. Use `PALLOC_VERBOSE=1` to check if palloc was successfully 
redirected.

### Other Platforms

You always link with `palloc.dll` but for different platforms you may 
need a specific redirection DLL:

- __x64__: `palloc-redirect.dll`.
- __x86__: `palloc-redirect32.dll`. Use for older 32-bit Windows programs.
- __arm64__: `palloc-redirect-arm64.dll`. Use for native Windows arm64 programs.
- __arm64ec__: `palloc-redirect-arm64ec.dll`. The [arm64ec] ABI is "emulation compatible" 
  mode on Windows arm64. Unfortunately we cannot run x64 code emulated on Windows arm64 with
  the x64 palloc override directly (since the C runtime always uses `arm64ec`). Instead:
  1. Build the program as normal for x64 and link as normal with the x64 
     `palloc.lib` export library.
  2. Now separately build `palloc.dll` in `arm64ec` mode and _overwrite_ your
     previous (x64) `palloc.dll` -- the loader can handle the mix of arm64ec
     and x64 code. Now use `palloc-redirect-arm64ec.dll` to match your new
     arm64ec `palloc.dll`. The main program stays as is and can be fully x64 
     or contain more arm64ec modules. At runtime, the arm64ec `palloc.dll` will
     run with native arm64 instructions while the rest of the program runs emulated x64.

[arm64ec]: https://learn.microsoft.com/en-us/windows/arm/arm64ec


### Minject

We cannot always re-link an executable with `palloc.dll`, and similarly, we 
cannot always ensure that the DLL comes first in the import table of the final executable.
In many cases though we can patch existing executables without any recompilation
if they are linked with the dynamic C runtime (`ucrtbase.dll`) -- just put the 
`palloc.dll` into the import table (and put `palloc-redirect.dll` in the same 
directory) Such patching can be done for example with [CFF Explorer](https://ntcore.com/?page_id=388).

The `minject` program can also do this from the command line
Use `minject --help` for options:

```
> minject --help

minject:
  Injects the palloc dll into the import table of a 64-bit executable,
  and/or ensures that it comes first in het import table.

usage:
  > minject [options] <exe>

options:
  -h   --help        show this help
  -v   --verbose     be verbose
  -l   --list        only list imported modules
  -i   --inplace     update the exe in-place (make sure there is a backup!)
  -f   --force       always overwrite without prompting
       --postfix=<p> use <p> as a postfix to the palloc dll.
                     e.g. use --postfix=debug to link with palloc-debug.dll

notes:
  Without '--inplace' an injected <exe> is generated with the same name ending in '-mi'.
  Ensure 'palloc-redirect.dll' is in the same folder as the palloc dll.

examples:
  > minject --list myprogram.exe
  > minject --force --inplace myprogram.exe
```  

For x86 32-bit binaries, use `minject32`, and for arm64 binaries use `minject-arm64`.

