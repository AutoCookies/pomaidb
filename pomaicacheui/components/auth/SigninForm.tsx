"use client";

import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useRouter } from "next/navigation";
import Link from "next/link";
// Sử dụng Remix Icon cho Clean & Modern UI
import {
    RiLoader4Line,
    RiLockPasswordLine,
    RiMailLine,
    RiArrowRightLine,
    RiGithubFill,
    RiGoogleFill
} from "@remixicon/react";

import { useAuth } from "@/hooks/useAuth";
import { signinSchema, SigninValues } from "@/lib/validations/auth";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { cn } from "@/lib/utils";

export function SigninForm() {
    // login function này sẽ gọi authServices.signin bên trong AuthContext
    const { login } = useAuth();
    const router = useRouter();
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const form = useForm<SigninValues>({
        resolver: zodResolver(signinSchema),
        defaultValues: { email: "", password: "" },
    });

    async function onSubmit(data: SigninValues) {
        setIsLoading(true);
        setError(null);
        try {
            // Gọi hàm login (đã được bọc logic gọi API signin)
            await login(data);

            // Đăng nhập thành công -> Chuyển hướng Dashboard
            router.push("/dashboard");
        } catch (err: any) {
            // Hiển thị lỗi từ Backend (authServices ném ra)
            setError(err.message || "Đăng nhập thất bại. Vui lòng thử lại.");
        } finally {
            setIsLoading(false);
        }
    }

    return (
        <div className="grid gap-6">
            <Form {...form}>
                <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">

                    {/* Hiển thị thông báo lỗi nếu có */}
                    {error && (
                        <Alert variant="destructive" className="bg-red-900/10 border-red-900/20 text-red-500">
                            <AlertDescription>{error}</AlertDescription>
                        </Alert>
                    )}

                    {/* Email Input */}
                    <FormField
                        control={form.control}
                        name="email"
                        render={({ field }) => (
                            <FormItem>
                                <FormLabel className="text-xs uppercase text-muted-foreground font-semibold tracking-wider">Email</FormLabel>
                                <FormControl>
                                    <div className="relative group">
                                        <RiMailLine className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground transition-colors group-focus-within:text-pomai-gold" />
                                        <Input
                                            placeholder="user@pomai.cache"
                                            className="pl-9 bg-secondary/20 border-border focus-visible:ring-pomai-gold focus-visible:border-pomai-gold transition-all"
                                            {...field}
                                        />
                                    </div>
                                </FormControl>
                                <FormMessage />
                            </FormItem>
                        )}
                    />

                    {/* Password Input */}
                    <FormField
                        control={form.control}
                        name="password"
                        render={({ field }) => (
                            <FormItem>
                                <div className="flex items-center justify-between">
                                    <FormLabel className="text-xs uppercase text-muted-foreground font-semibold tracking-wider">Password</FormLabel>
                                    <Link
                                        href="/forgot-password"
                                        className="text-xs text-pomai-gold hover:text-pomai-red transition-colors hover:underline"
                                    >
                                        Forgot password?
                                    </Link>
                                </div>
                                <FormControl>
                                    <div className="relative group">
                                        <RiLockPasswordLine className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground transition-colors group-focus-within:text-pomai-gold" />
                                        <Input
                                            type="password"
                                            placeholder="••••••••"
                                            className="pl-9 bg-secondary/20 border-border focus-visible:ring-pomai-gold focus-visible:border-pomai-gold transition-all"
                                            {...field}
                                        />
                                    </div>
                                </FormControl>
                                <FormMessage />
                            </FormItem>
                        )}
                    />

                    {/* Submit Button */}
                    <Button
                        type="submit"
                        disabled={isLoading}
                        className={cn(
                            "w-full py-6 font-bold text-white transition-all shadow-lg",
                            "bg-pomai-red hover:bg-pomai-red/90",
                            "shadow-pomai-red/20 hover:shadow-pomai-red/40"
                        )}
                    >
                        {isLoading ? (
                            <RiLoader4Line className="mr-2 h-5 w-5 animate-spin" />
                        ) : (
                            <>
                                Access System <RiArrowRightLine className="ml-2 h-4 w-4" />
                            </>
                        )}
                    </Button>
                </form>
            </Form>

            {/* Divider */}
            <div className="relative">
                <div className="absolute inset-0 flex items-center">
                    <span className="w-full border-t border-border" />
                </div>
                <div className="relative flex justify-center text-xs uppercase">
                    <span className="bg-background px-2 text-muted-foreground">Or continue with</span>
                </div>
            </div>

            {/* Social Buttons */}
            <div className="grid grid-cols-2 gap-4">
                <Button variant="outline" className="w-full bg-background hover:bg-secondary/50 border-border hover:text-pomai-gold transition-colors">
                    <RiGoogleFill className="mr-2 h-4 w-4" /> Google
                </Button>
                <Button variant="outline" className="w-full bg-background hover:bg-secondary/50 border-border hover:text-pomai-gold transition-colors">
                    <RiGithubFill className="mr-2 h-4 w-4" /> GitHub
                </Button>
            </div>

            <div className="text-center text-sm text-muted-foreground">
                New to Pomai?{" "}
                <Link href="/signup" className="font-medium text-pomai-gold hover:text-pomai-red hover:underline transition-colors">
                    Initialize Account
                </Link>
            </div>
        </div>
    );
}