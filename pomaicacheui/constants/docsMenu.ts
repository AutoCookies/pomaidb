/* constants/docMenu.ts
   Định nghĩa cấu trúc menu tài liệu với phân level rõ ràng.
*/

export type DocMenuItem = {
    id: string
    title: string
    href?: string
    // level: 1 = top level, 2 = sub, 3 = sub-sub, ...
    level: number
    children?: DocMenuItem[]
}

export const docMenu: DocMenuItem[] = [
    {
        id: "installation",
        title: "Installation",
        href: "/documents/installation",
        level: 1
    },
    {
        id: "quickstart",
        title: "Quickstart",
        href: "/documents/quickstart",
        level: 1
    },
    {
        id: "operations",
        title: "API Operations",
        href: "/documents/operations",
        level: 1,
        // children: [
        //     {
        //         id: "endpoints",
        //         title: "Các endpoints",
        //         href: "/intro/api/endpoints",
        //         level: 2,
        //         children: [
        //             {
        //                 id: "put",
        //                 title: "PUT",
        //                 href: "/intro/api/endpoints#put",
        //                 level: 3,
        //             },
        //             {
        //                 id: "get",
        //                 title: "GET",
        //                 href: "/intro/api/endpoints#get",
        //                 level: 3,
        //             },
        //             {
        //                 id: "delete",
        //                 title: "DELETE",
        //                 href: "/intro/api/endpoints#delete",
        //                 level: 3,
        //             },
        //         ],
        //     },
        //     {
        //         id: "auth",
        //         title: "Xác thực (API keys)",
        //         href: "/intro/api/auth",
        //         level: 2,
        //     },
        //     {
        //         id: "ttl",
        //         title: "TTL & Expiration",
        //         href: "/intro/api/ttl",
        //         level: 2,
        //     },
        // ],
    },
]